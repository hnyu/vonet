# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import matplotlib.cm
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

import alf
import alf.summary.render as render
from alf.algorithms.agent_helpers import AgentHelper
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import AlgStep, LossInfo, StepType, namedtuple
from alf.utils import losses, tensor_utils
from alf.utils.averager import ScalarWindowAverager
from alf.utils.summary_utils import record_time
from alf.summary.summary_ops import _summary_writer_stack
from alf.metrics import metric
from vonet.vonet_algorithm import VONetAlgorithm

AgentState = namedtuple(
    'AgentState', ['vonet', 'vi', 'seg', 'gt_seg', 'steps'], default_value=())

AgentInfo = namedtuple(
    'AgentInfo', [
        'vonet', 'vi', 'seg', 'gt_seg', 'video', 'fg_ari', 'ari', 'miou',
        'slots', 'prior_rec'
    ],
    default_value=())


class History(
        namedtuple("History", ['z', 'kld', 'rec_loss'], default_value=None)):
    def update(self, step):
        return alf.nest.map_structure_up_to(
            self, lambda h, s: torch.cat(
                [h[:, 1:, ...], s.unsqueeze(1)], dim=1) if s != () else h,
            self, step)

    @classmethod
    def spec(cls, T, spec_mapping={}):
        def _create_spec(path):
            shape = (T, )
            dtype = torch.float32
            if path in spec_mapping:
                spec = spec_mapping[path]
                shape += spec.shape
                dtype = spec.dtype
            return alf.TensorSpec(shape, dtype=dtype)

        return alf.nest.py_map_structure_with_path(
            lambda path, _: _create_spec(path), cls())


import random
colors = matplotlib.cm.tab20b.colors
colors = list(colors) + list(matplotlib.cm.tab20c.colors)
random.shuffle(colors)
colors = torch.tensor([(0, 0, 0)] + colors)  # len=41


def _render_segmentation(mat):
    """ merge all masks into one image """
    # mat: [...,H,W]
    # [...,H,W,3]
    rendered = colors[mat.to(torch.int64)]
    dims = list(range(rendered.ndim - 3)) + [-1, -3, -2]
    # [...,3,H,W]
    return rendered.permute(dims)


def _overlay_segmentation_on_image(seg, image):
    np_colors = np.round(colors.cpu().numpy() * 255).astype(np.uint8)
    image = np.round(image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    num_masks = np.max(seg) + 1
    # Create a copy of the RGB image to overlay the current mask
    overlay_image = image.copy()
    for mask_id in range(num_masks):
        # Extract the mask for the current mask_id
        mask = (seg == mask_id).astype(np.uint8)
        # Set the color for the current mask
        color = np_colors[mask_id]
        # Overlay the mask with transparency
        overlay_image[mask > 0] = (
            (1 - 0.4) * overlay_image[mask > 0] + 0.4 * color).astype(np.uint8)
        # Draw a boundary around the current mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (255, 255, 255), 1)
    return overlay_image


class Visualizer(Algorithm):
    def __init__(self, z_spec, slots, T=24, name="Visualizer"):
        G = slots
        state_spec = History.spec(
            T, {
                'z': alf.TensorSpec((G, ) + z_spec.shape[1:]),
                'kld': alf.TensorSpec((G, ))
            })
        super(Visualizer, self).__init__(
            predict_state_spec=state_spec, name=name)
        self._slots = slots
        plt.style.use('seaborn-pastel')

    def _tensor_to_image(self, tensor, name, height=None, width=None):
        if isinstance(tensor, torch.Tensor):
            # Convert a [0,1] tensor to uint8 image
            img = torch.round(tensor * 255).to(torch.uint8)
            if img.ndim == 3:
                img = img.permute((1, 2, 0))
            img = render.Image(img.cpu().numpy())
        else:
            img = render.Image(tensor)
        if height or width:
            img.resize(height, width)
        text = render.render_text(
            name='', data=name, img_height=10, fig_height=0.1, font_size=8)
        return render.Image.stack_images([img, text], horizontal=False)

    def _image(self, tensor, name, height=256):
        return self._tensor_to_image(tensor, name, height=height)

    def predict_step(self, inputs, state):
        alg_step, time_step = inputs
        z = alg_step.output.z  # [B,G,D]
        new_state = state

        info = alg_step.info.vonet
        im = time_step.observation['observation'].squeeze(0)
        video_id = int(time_step.env_info['video_id'].squeeze(0).cpu().numpy())
        frame_id = int(time_step.env_info['frame_id'].squeeze(0).cpu().numpy())

        imgs = OrderedDict()
        if render.is_rendering_enabled():
            new_state = new_state.update(
                History(
                    z=z.detach(),
                    kld=info.vae.slot_kld.detach(),
                    rec_loss=info.dec.rec_loss.detach()))

            rec = info.dec.pred_rec.squeeze(0)
            prior_rec = alg_step.info.prior_rec.squeeze(0)
            imgs['input'] = self._image(im, f'vid/{video_id}_{frame_id}')
            imgs['rec'] = self._image(rec, 'rec')
            imgs['prior_rec'] = self._image(prior_rec, 'prior_rec')

            # [...,H,W]; uint8
            gt_seg = alg_step.info.gt_seg.squeeze(-3)
            fg_seg = gt_seg.to(torch.bool) * alg_step.info.seg
            fg_seg = _render_segmentation(fg_seg)
            seg = _render_segmentation(alg_step.info.seg)
            gt_seg = _render_segmentation(gt_seg)
            overlay_seg = _overlay_segmentation_on_image(
                alg_step.info.seg[0].cpu().numpy(),
                im.cpu().numpy())
            imgs['segmentations'] = render.Image.stack_images(
                (self._image(seg[0, ...], 'seg', height=256),
                 self._image(overlay_seg, 'overlay_seg', height=256),
                 self._image(fg_seg[0, ...], 'fg_seg', height=256),
                 self._image(gt_seg[0, ...], 'gt_seg', height=256)))

            kld_curve = new_state.kld[..., 0]  # pick the first slot
            mask = info.mask[0, 0, 0, ...].exp()  # [B,G,1,H,W] -> [H,W]
            imgs['mask'] = self._image(mask, 'slot0_mask')
            imgs['kld_curve'] = render.render_curve(
                name='Slot0',
                data=kld_curve,
                img_height=128 * 3,
                img_width=1000,
                dpi=500,
                x_label='Time step',
                y_label='KLD per dim',
                linewidth=2,
                color='blue',
                x_ticks=np.arange(kld_curve.shape[-1]),
                figsize=(6, 2))

        return AlgStep(state=new_state, info=imgs)


def get_segmentation(mask, threshold):
    """Take the argmax+1 of mask prob as the seg id. If the max prob is below
    the threshold, then that pixel is marked as 0.
    """
    # mask: [B,G,1,H,W], log space
    # [B,G,H,W]
    mask = mask.squeeze(2).exp()
    # [B,1,H,W]
    seg = torch.argmax(mask, dim=1, keepdim=True)
    # [B,1,H,W]
    prob = torch.gather(mask, dim=1, index=seg)
    fg = (prob > threshold).to(torch.int32)
    seg = (seg + 1) * fg
    # [B,H,W]
    return seg.squeeze(1)


def compute_ari(seg,
                gt_seg,
                num_groups,
                temporal_padding,
                ignore_background=True):
    """Converted from the JAX version:
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py#L111

    Args:
        temporal_padding: a mask of shape [B,T] indicating which timesteps are
            invalid but just paddings.
    """
    # Following SAVI, it may be that num_groups <= max(segmentation). We prune
    # out extra objects here. For example, movi_e has an instance id of 64 in an
    # outlier video.
    # https://github.com/google-research/slot-attention-video/blob/main/savi/lib/preprocessing.py#L414
    gt_seg = torch.where(gt_seg >= num_groups, torch.zeros_like(gt_seg),
                         gt_seg)

    seg = torch.nn.functional.one_hot(seg.to(torch.int64), num_groups).float()
    gt_seg = torch.nn.functional.one_hot(gt_seg.to(torch.int64),
                                         num_groups).float()

    # [B,T,1,1,1]
    temporal_padding = temporal_padding.reshape(*temporal_padding.shape, 1, 1,
                                                1)
    gt_seg = gt_seg * temporal_padding

    if ignore_background:
        # remove background (id=0).
        gt_seg = gt_seg[..., 1:]

    N = torch.einsum('bthwc,bthwk->bck', gt_seg, seg)  # [B,c,k]
    A = N.sum(-1)  # row-sum  [B,c]
    B = N.sum(-2)  # col-sum  [B,k]
    num_points = A.sum(1)  # [B]

    rindex = (N * (N - 1)).sum((1, 2))  # [B]
    aindex = (A * (A - 1)).sum(1)  # [B]
    bindex = (B * (B - 1)).sum(1)  # [B]

    expected_rindex = aindex * bindex / torch.clamp(
        num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def compute_mIoU(seg, gt_seg, temporal_padding):
    # seg: [B,T,H,W]
    B = gt_seg.shape[0]
    # [B,T,1,1,1]
    temporal_padding = temporal_padding.reshape(*temporal_padding.shape, 1, 1,
                                                1)

    def _compress_ids(x, debug=False):
        # x: [T,H,W]
        x = x.to(torch.int64)
        ids = torch.unique(x)
        if debug:
            print(ids)
        id_mapping = torch.zeros((ids.max() + 1, ), dtype=torch.int64)
        # mapping: id i,j,k.. -> index 0,1,2...
        id_mapping[ids] = torch.arange(ids.numel(), dtype=torch.int64)
        return id_mapping[x], ids.numel()

    def _count(x, y, padding):
        return torch.einsum('thwc,thwk->ck', x * padding, y)

    cost = []
    for b in range(B):
        # [T,H,W]: [0,1,2,...]
        gt_seg_i, N = _compress_ids(gt_seg[b])
        seg_i, M = _compress_ids(seg[b])
        gt_seg_i = torch.nn.functional.one_hot(gt_seg_i, N).float()
        seg_i = torch.nn.functional.one_hot(seg_i, M).float()
        # [T,1,1,1]
        padding_i = temporal_padding[b]

        intersection = _count(gt_seg_i, seg_i, padding_i)
        all_ = _count(
            torch.ones_like(gt_seg_i), torch.ones_like(seg_i), padding_i)
        union = all_ - _count(1 - gt_seg_i, 1 - seg_i, padding_i)
        mIoU_mat = intersection / (union + 1e-8)

        np_cost_mat = (-mIoU_mat).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(np_cost_mat)
        cost.append(np_cost_mat[row_ind, col_ind].sum() / N)

    return -torch.tensor(cost).float()


class VONetAgent(RLAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=(),
                 env=None,
                 config=None,
                 optimizer=None,
                 debug_summaries=False,
                 name="VONetAgent"):
        agent_helper = AgentHelper(AgentState)

        assert isinstance(observation_spec, dict)

        obs_spec = observation_spec['observation']
        vonet = VONetAlgorithm(input_tensor_spec=obs_spec)
        agent_helper.register_algorithm(vonet, 'vonet')

        vis = Visualizer(vonet.z_spec, vonet.slots)
        agent_helper.register_algorithm(vis, 'vi')

        state_specs = agent_helper.state_specs()
        seg_spec = alf.TensorSpec(
            (24, ) + obs_spec.shape[-2:], dtype=torch.uint8)
        state_specs['predict_state_spec'] = state_specs[
            'predict_state_spec']._replace(
                steps=alf.TensorSpec(()),
                # [T,H,W]
                seg=seg_spec,
                gt_seg=seg_spec)
        state_specs['rollout_state_spec'] = state_specs['predict_state_spec']

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            optimizer=optimizer,
            is_on_policy=False,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name,
            **state_specs)

        self._vonet = vonet
        self._visualizer = vis
        self._fg_ari_averager = ScalarWindowAverager(window_size=100)
        self._ari_averager = ScalarWindowAverager(window_size=100)
        self._miou_averager = ScalarWindowAverager(window_size=100)

        dataset = alf.get_config_value('_CONFIG._USER.dataset')
        if 'movi' in dataset:
            self._num_groups = 25
        else:
            self._num_groups = vonet.slots + 1

    def _step(self, time_step, state):
        obs = time_step.observation
        vonet_step = self._vonet.train_step(obs['observation'], state.vonet)
        info = AgentInfo(vonet=vonet_step.info)
        state = state._replace(vonet=vonet_step.state)
        alg_step = AlgStep(output=vonet_step.output, state=state, info=info)
        # assemble a segmentation image from the masks
        alg_step = alf.nest.set_field(
            alg_step, 'info.seg',
            get_segmentation(alg_step.info.vonet.mask, threshold=0.3))
        alg_step = alf.nest.set_field(alg_step, 'info.gt_seg',
                                      time_step.observation["segmentation"])
        return alg_step

    def train_step(self, time_step, state, rollout_info=None):
        alg_step = self._step(time_step, state)
        alg_step = alf.nest.set_field(alg_step, 'info.video',
                                      time_step.observation['observation'])
        alg_step = alf.nest.set_field(alg_step, 'info.slots',
                                      alg_step.output.z)
        return alg_step._replace(output=())

    def _ari_step(self, time_step, alg_step):
        state = alg_step.state
        steps = state.steps + 1
        # compute ARI
        seg = torch.cat(
            (alg_step.info.seg.unsqueeze(1), state.seg[:, :-1, ...]), dim=1)
        gt_seg = torch.cat((alg_step.info.gt_seg, state.gt_seg[:, :-1, ...]),
                           dim=1)

        temporal_padding = (tensor_utils.tensor_extend_new_dim(
            torch.arange(seg.shape[1]), dim=0, n=seg.shape[0]) <
                            steps.unsqueeze(-1)).to(torch.int32)
        fg_ari = compute_ari(seg, gt_seg, self._num_groups, temporal_padding,
                             True)  # [B]
        ari = compute_ari(seg, gt_seg, self._num_groups, temporal_padding,
                          False)  # [B]
        with record_time('time/compute_mIoU'):
            miou = compute_mIoU(seg, gt_seg, temporal_padding)
        alg_step = alf.nest.set_field(alg_step, 'info.fg_ari', fg_ari)
        alg_step = alf.nest.set_field(alg_step, 'info.ari', ari)
        alg_step = alf.nest.set_field(alg_step, 'info.miou', miou)
        alg_step = alf.nest.set_field(alg_step, 'state.seg', seg)
        alg_step = alf.nest.set_field(alg_step, 'state.gt_seg', gt_seg)
        alg_step = alf.nest.set_field(alg_step, 'state.steps', steps)
        return alg_step

    def predict_step(self, time_step, state):
        alg_step = self._step(time_step, state)
        step = self._vonet.reconstruct_with_prior(
            time_step.observation['observation'], state.vonet)
        alg_step = alf.nest.set_field(alg_step, 'info.prior_rec',
                                      step.info.dec.pred_rec)
        alg_step = self._ari_step(time_step, alg_step)

        action = self._action_spec.sample(time_step.reward.shape[:1])
        vi_step = self._visualizer.predict_step((alg_step, time_step),
                                                state.vi)
        alg_step = alf.nest.set_field(alg_step, 'info.vi', vi_step.info)
        alg_step = alf.nest.set_field(alg_step, 'state.vi', vi_step.state)
        return alg_step._replace(output=action)

    def rollout_step(self, time_step, state):
        alg_step = self._step(time_step, state)
        alg_step = self._ari_step(time_step, alg_step)
        fg_ari = alg_step.info.fg_ari[time_step.is_last()]
        ari = alg_step.info.ari[time_step.is_last()]
        miou = alg_step.info.miou[time_step.is_last()]
        if fg_ari.numel() > 0:
            self._fg_ari_averager.update(fg_ari)
        if ari.numel() > 0:
            self._ari_averager.update(ari)
        if miou.numel() > 0:
            self._miou_averager.update(miou)
        action = self._action_spec.sample(time_step.reward.shape[:1])
        # Remove info so that it won't waste replay memory
        return alg_step._replace(output=action, info=())

    def calc_loss(self, info: AgentInfo):
        with alf.summary.scope(self._name):
            alf.summary.scalar("rollout_fg_ari", self._fg_ari_averager.get())
            alf.summary.scalar("rollout_ari", self._ari_averager.get())
            alf.summary.scalar("rollout_mIoU", self._miou_averager.get())
            if alf.summary.should_record_summaries():
                vis_n = 4
                gt_seg = info.gt_seg[:, :vis_n, 0, ...]
                seg = info.seg[:, :vis_n, ...]
                fg_seg = gt_seg.to(
                    torch.bool) * seg  # only show the foreground seg
                # [T,vis_n,3,H,W]
                gt_seg = _render_segmentation(gt_seg)
                seg = _render_segmentation(seg)
                fg_seg = _render_segmentation(fg_seg)
                # [T,vis_n,G,1,H,W]
                mask = info.vonet.mask.exp()[:, :vis_n, ...]
                # [T,vis_n,C,H,W]
                video = info.video[:, :vis_n, ...]

                writer = _summary_writer_stack[-1]
                writer.add_video("input/rgb", video.transpose(0, 1))

                def _show_rec(rec, name):
                    # rec: [vis_n,T,C,H,W]
                    writer.add_video(name, rec.transpose(0, 1))

                if True:
                    # compute autogresstive reconstructed video
                    # [T,vis_n,G,D]
                    slots = info.slots[:, :vis_n, ...]
                    bs = alf.layers.BatchSquash(2)
                    # [T*vis_n,C,H,W]
                    with record_time('time/autoregressive_decode'):
                        auto_rec = self._vonet._decoder.autoregressive_decode(
                            bs.flatten(slots))
                    rec = bs.unflatten(auto_rec)
                    _show_rec(rec, 'autoregress_rec')

                if info.vonet.dec.rec != ():
                    _show_rec(info.vonet.dec.rec[:, :vis_n, ...], 'vae_rec')
                _show_rec(info.vonet.dec.pred_rec[:, :vis_n, ...], 'pred_rec')

                for i in range(vis_n):
                    writer.add_video(f"mask/{i}", mask[:, i, ...].transpose(
                        0, 1))

                writer.add_video('seg/gt', gt_seg.transpose(0, 1))
                writer.add_video('seg/pred', seg.transpose(0, 1))
                writer.add_video('seg/pred_fg', fg_seg.transpose(0, 1))

                # show slot kld [T,vis_n,G]
                alf.summary.render.enable_rendering()
                slot_kld = info.vonet.vae.slot_kld[:, :vis_n, ...].detach()
                for i in range(vis_n):
                    slot_kld_img = alf.summary.render.render_curve(
                        name="slot_kld",
                        legends=[
                            f'slot{j}' for j in range(slot_kld.shape[-1])
                        ],
                        data=slot_kld[:, i, ...].transpose(0, 1),
                        figsize=(3, 3))
                    writer.add_image(
                        tag=f'slot_kld/{i}',
                        img_tensor=slot_kld_img.data,
                        dataformats='HWC')

        return self._vonet.calc_loss(info.vonet)
