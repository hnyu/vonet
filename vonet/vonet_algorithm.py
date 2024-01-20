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

from functools import partial
from typing import Callable, Tuple, Union, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.monet_algorithm import MoNetUNet
from alf.algorithms.vae import VAEOutput
from alf.data_structures import namedtuple, AlgStep, LossInfo
from alf.utils import tensor_utils, dist_utils, conditional_ops, losses, common
from alf.utils.schedulers import Scheduler, as_scheduler
from vonet.transformer_decoder import DiscreteTransformerDecoder, _gumbel_softmax


def make_gaussian(z_mean_and_log_var):
    D = z_mean_and_log_var.shape[-1] // 2
    z_mean = z_mean_and_log_var[..., :D]
    z_log_var = z_mean_and_log_var[..., D:]
    # [B,G,D]
    return td.Independent(
        td.Normal(loc=z_mean, scale=z_log_var.exp()),
        reinterpreted_batch_ndims=1)


def create_transformer(blocks,
                       heads,
                       d_model,
                       memory_size,
                       position_encoding='none',
                       dropout=0.):
    trans_blocks = []
    for i in range(blocks):
        trans_blocks.append(
            alf.layers.TransformerBlock(
                d_model=d_model,
                num_heads=heads,
                memory_size=memory_size,
                d_k=d_model,
                d_v=d_model,
                d_ff=d_model * 2,
                dropout=dropout,
                positional_encoding=position_encoding))
    return nn.Sequential(*trans_blocks)


class PositionEmbeddingNet(alf.networks.Network):
    def __init__(self, input_tensor_spec, name="PositionEmbeddingNet"):
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)
        hidden_size = input_tensor_spec.shape[0]
        self._proj = nn.Sequential(
            alf.layers.Conv2D(4, hidden_size, 1),
            alf.layers.Conv2D(hidden_size, input_tensor_spec.shape[0], 1,
                              alf.math.identity))
        y = torch.linspace(0., 1., steps=input_tensor_spec.shape[-2])
        x = torch.linspace(0., 1., steps=input_tensor_spec.shape[-1])
        yy, xx = torch.meshgrid(y, x)
        # [1,H,W]
        yy, xx = yy.unsqueeze(0), xx.unsqueeze(0)
        grid = torch.cat([yy, xx, 1. - yy, 1. - xx], dim=0)
        # [1,4,H,W]
        self._grid = grid.unsqueeze(0)

    def forward(self, img, state=()):
        # [1,C,H,W]
        pos_embedding = self._proj(self._grid)
        return img + pos_embedding, state


SlotVAEInfo = namedtuple(
    "SlotVAEInfo", ["kld", "slot_kld", "vae", "z_prior"], default_value=())


@alf.configurable
class SequentialSlotVAE(Algorithm):
    """A temporal VAE for slots.
    """

    def __init__(self,
                 slot_size: int,
                 slots: int,
                 slots_interaction_iters: int = 3,
                 slots_interaction_heads: int = 3,
                 beta: Union[float, Scheduler] = 1.,
                 alpha: float = 0.5,
                 name: str = "SequentialSlotVAE"):
        D = slot_size
        H = slot_size
        G = slots

        super(SequentialSlotVAE, self).__init__(name=name)

        self._alpha = alpha
        self._beta = as_scheduler(beta)

        # if slots_interaction_iters=0, each slot predicts itself independently
        layers = [alf.layers.FC(D, H)]
        if slots_interaction_iters > 0:
            layers += [
                create_transformer(
                    blocks=slots_interaction_iters,
                    heads=slots_interaction_heads,
                    d_model=H,
                    memory_size=G)
            ]
        layers += [alf.layers.FC(H, H, torch.relu_), alf.layers.FC(H, 2 * D)]
        self._predictor = nn.Sequential(*layers)
        self._slot_posterior = nn.Sequential(
            alf.layers.FC(D, D, torch.relu_), alf.layers.FC(D, 2 * D))

    def _compute_kl(self, p, q):
        if self._alpha == 0.5:
            # [B,G]
            kld = td.kl.kl_divergence(p, q)
        else:
            dp = common.detach(p)
            dq = common.detach(q)
            kld_dp_q = td.kl.kl_divergence(dp, q)
            kld_p_dq = td.kl.kl_divergence(p, dq)
            # [B,G]
            kld = self._alpha * kld_dp_q + (1 - self._alpha) * kld_p_dq
        # kld is per-dimension
        return kld / np.prod(p.event_shape)  # [B,G] or [B,G-1]

    def _compute_prior_dist(self, slots):
        # [B,G,2*D]
        prior_z_mean_and_log_var = self._predictor(slots)
        prior_dist = make_gaussian(prior_z_mean_and_log_var)
        return prior_dist

    def _compute_posterior_dist(self, new_slots):
        z_mean_and_log_var = self._slot_posterior(new_slots)
        posterior_dist = make_gaussian(z_mean_and_log_var)
        return posterior_dist

    def train_step(self, inputs, state=()):
        new_slots, slots, kld_mask = inputs
        prior_dist = self._compute_prior_dist(slots)
        dist = self._compute_posterior_dist(new_slots)
        slot_kld = self._compute_kl(dist, prior_dist)  # [B,G]
        kld = slot_kld.sum(-1) * kld_mask
        with torch.no_grad():
            # For visualization only
            z_prior = prior_dist.rsample()
        return AlgStep(
            output=VAEOutput(
                z=dist.rsample(), z_mode=alf.utils.dist_utils.get_mode(dist)),
            info=SlotVAEInfo(kld=kld, slot_kld=slot_kld, z_prior=z_prior))

    def calc_loss(self, info: SlotVAEInfo):
        loss = info.kld * self._beta()
        with alf.summary.scope(self.name):
            alf.summary.histogram('kld/value', info.kld)
            alf.summary.scalar('beta', self._beta())
        return LossInfo(loss=loss, extra=info.kld)


VONetState = namedtuple("VONetState", ['steps', 'slots'], default_value=())
VONetInfo = namedtuple(
    "VONetInfo", ['mask', 'prior_mask', 'vae', 'dec'], default_value=())


@alf.configurable
class VONetAlgorithm(Algorithm):
    def __init__(self,
                 n_slots: int,
                 slot_size: int,
                 input_tensor_spec: alf.NestedTensorSpec,
                 mask_transformer_layers: int = 3,
                 mask_transformer_heads: int = 3,
                 attention_unet_cls: Callable = MoNetUNet,
                 encoder_cls: Callable = alf.networks.EncodingNetwork,
                 decoder_cls: Callable = DiscreteTransformerDecoder,
                 init_slot_transformer_layers: int = 3,
                 init_slot_transformer_heads: int = 3,
                 vae_start_steps: int = 0,
                 no_unet_refinement: bool = False,
                 name: str = "VONetAlgorithm"):

        _, H, W = input_tensor_spec.shape
        encoding_size = slot_size

        # only take rgb as the input
        input_tensor_spec = alf.TensorSpec(
            shape=(3, H, W), dtype=input_tensor_spec.dtype)

        enc = encoder_cls(
            input_tensor_spec=input_tensor_spec, last_filter=encoding_size)
        _, h, w = enc.output_spec.shape
        # self._encoder will downsize the image. All subsequent convolutions
        # will be performed on hxw.
        # Because of downsizing, we need to upscale the eventual mask
        mask_resize = lambda m: F.interpolate(
            m.squeeze(2), scale_factor=H // h).unsqueeze(2)

        super().__init__(
            train_state_spec=VONetState(
                slots=alf.TensorSpec((n_slots, slot_size)),
                steps=alf.TensorSpec((), dtype=torch.int64)),
            name=name)

        self._n_slots = n_slots
        self._slot_size = slot_size

        self._mask_resize = mask_resize
        self._context_dim = encoding_size

        self._init_slots_transformer = create_transformer(
            init_slot_transformer_layers, init_slot_transformer_heads,
            slot_size, n_slots)

        # This layer projects the slot encoding into a context embedding to be
        # conditioned on by the encoder at the next step
        self._context_layer = nn.Sequential(
            alf.layers.FC(slot_size, 2 * slot_size, torch.relu_),
            alf.layers.FC(2 * slot_size, self._context_dim))

        self._encoder = alf.nn.Sequential(
            enc, PositionEmbeddingNet(enc.output_spec))

        self._img_key_proj = alf.layers.Conv2D(
            encoding_size,
            encoding_size,
            kernel_size=1,
            activation=alf.math.identity)
        self._img_val_proj = alf.layers.Conv2D(
            encoding_size,
            encoding_size,
            kernel_size=1,
            activation=alf.math.identity)

        def _make_additive_net(img_spec, channels):
            c = img_spec.shape[0]
            parallel = (alf.math.identity, )
            spec = (img_spec, )
            for k in channels:
                parallel = parallel + (alf.layers.Conv2D(
                    k, c, 3, padding=1, activation=alf.math.identity), )
                spec = spec + (alf.TensorSpec((k, h, w)), )
            return alf.nn.Sequential(
                alf.nn.Parallel(parallel, input_tensor_spec=spec),
                alf.nest.utils.NestSum())

        unet_input_spec = self._encoder.output_spec
        self._attention_preproc_net = _make_additive_net(
            unet_input_spec, (2, self._context_dim))
        self._attention_net = attention_unet_cls(
            input_tensor_spec=unet_input_spec, output_channels=1)

        K = self._attention_net.encoding_dim
        self._mask_encoding_transformer = create_transformer(
            mask_transformer_layers, mask_transformer_heads, K, n_slots)

        self._decoder = decoder_cls(n_slots=n_slots, slot_size=slot_size)

        self._slot_gru_layer = nn.GRUCell(
            input_size=encoding_size, hidden_size=slot_size)
        self._slot_mlp = nn.Sequential(
            alf.layers.FC(slot_size, 2 * slot_size, torch.relu_),
            alf.layers.FC(2 * slot_size, slot_size))
        self._slot_ln = nn.LayerNorm(slot_size)

        self._slot_vae = SequentialSlotVAE(slot_size, n_slots)
        self._vae_start_steps = vae_start_steps

        self._no_unet_refinement = no_unet_refinement

    @property
    def z_spec(self):
        return alf.TensorSpec((self._n_slots, self._slot_size))

    @property
    def slots(self):
        return self._n_slots

    def _init_slots(self, batch_size, dtype):
        """Initialize prior slots for a first frame by
        1. Sampling ``n_slots`` noise vectors from :math:`\mathcal{N}(0,1)`.
        2. Forward the noise vectors through a transformer to get the randomized
           slots history.
        """
        noise = torch.randn((batch_size, self._n_slots,
                             self._slot_size)).to(dtype)
        return self._init_slots_transformer(noise)

    def _calc_encoding(self, enc_img, mask_logprobs):
        # [B,G,1,h,w]
        mask_probs = mask_logprobs.exp()
        img_val = self._img_val_proj(enc_img)
        img_val = img_val.unsqueeze(1)
        # [B,G,c,h,w]
        masked_img = img_val * mask_probs
        # [B,G,c]
        encodings = masked_img.mean((-2, -1))
        return encodings

    def _encoder_step(self, img, state):
        # [B,G,c]
        reset = (state.steps == 0)
        # Update the context if reset==True
        slots = conditional_ops.conditional_update(
            target=state.slots,
            cond=reset,
            func=lambda r: self._init_slots(
                batch_size=r.shape[0], dtype=state.slots.dtype),
            r=reset)
        context = self._context_layer(slots)

        # [B,c,h,w]
        enc_img = self._encoder(img)[0]

        # mask: [B,G,1,h,w]
        mask_logprobs, prior_mask = self._compute_mask_logprobs(
            enc_img, context)

        # The masks in ``info`` is used as the final mask output
        info = VONetInfo(mask=mask_logprobs, prior_mask=prior_mask)
        # We need to resize the mask to the original image dimension for H=128
        info = alf.nest.map_structure(self._mask_resize, info)

        encodings = self._calc_encoding(enc_img, mask_logprobs)

        bs = alf.layers.BatchSquash(2)
        # s_t = gru(x_t,s_{t-1})
        new_slots = self._slot_gru_layer(
            bs.flatten(encodings), bs.flatten(slots))
        new_slots = bs.unflatten(new_slots)
        new_slots = new_slots + self._slot_mlp(new_slots)
        new_slots = self._slot_ln(new_slots)

        # z_t ~ posterior(z_t|s_t), z_t' ~ prior(z_t'|s_{t-1})
        kld_mask = (state.steps >= self._vae_start_steps).float()
        vae_step = self._slot_vae.train_step((new_slots, slots, kld_mask))
        return AlgStep(
            output=vae_step.output,
            state=VONetState(slots=new_slots, steps=state.steps + 1),
            info=info._replace(vae=vae_step.info))

    def _compute_mask_logprobs(self, img, context):
        ### Compute the foreground slots in parallel
        bs = alf.layers.BatchSquash(2)

        img_key = self._img_key_proj(img)
        img_key = img_key * (self._context_dim**(-0.5))
        # [B,G,h,w]
        prior_mask = torch.einsum("bchw,bgc->bghw", img_key, context)
        # [B,G,1,h,w]
        prior_mask = F.log_softmax(prior_mask.unsqueeze(2), dim=1)

        if self._no_unet_refinement:
            return prior_mask, prior_mask

        # [B*G,C,H,W]
        img = tensor_utils.tensor_extend_new_dim(img, dim=1, n=self._n_slots)
        # [B,G,C,h,w]
        context = alf.utils.tensor_utils.spatial_broadcast(context, img.shape)
        features, _ = self._attention_preproc_net(
            alf.nest.map_structure(
                bs.flatten,
                (img, torch.cat([prior_mask.exp(), prior_mask], dim=2),
                 context)))

        # [B*G,K]
        mask_encoding, inter_encodings = self._attention_net.encode(features)
        # [B,G,K]
        mask_encoding = self._mask_encoding_transformer(
            bs.unflatten(mask_encoding))
        # [B*G,1,H,W]
        mask_logits = self._attention_net.decode(
            bs.flatten(mask_encoding), inter_encodings)
        # [B,G,1,H,W]
        mask_logprobs = F.log_softmax(
            bs.unflatten(mask_logits) + prior_mask, dim=1)
        return mask_logprobs, prior_mask

    def _decoder_step(self, z, img, info):
        # z - [B,G,D]
        step = self._decoder.train_step((img, z))
        return info._replace(dec=step.info)

    def train_step(self, inputs: torch.Tensor, state=()):
        alg_step = self._encoder_step(inputs, state)
        info = self._decoder_step(alg_step.output.z, inputs, alg_step.info)
        return alg_step._replace(info=info)

    def reconstruct_with_prior(self, inputs: torch.Tensor, state=()):
        alg_step = self._encoder_step(inputs, state)
        info = alg_step.info
        info = self._decoder_step(info.vae.z_prior, inputs, info)
        return alg_step._replace(info=info)

    def calc_loss(self, info: VONetInfo):
        vae_loss = self._slot_vae.calc_loss(info.vae)
        dec_loss = self._decoder.calc_loss(info.dec)
        with alf.summary.scope(self.name):
            for i in range(self._n_slots):
                alf.summary.scalar(
                    'slot_mask/%d' % i,
                    (info.mask[:, :, i, ...].exp() > 0.3).float().mean())
        return LossInfo(
            loss=dec_loss.loss + vae_loss.loss,
            extra=VONetInfo(dec=dec_loss.extra, vae=vae_loss.extra))
