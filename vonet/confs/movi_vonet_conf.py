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
import numpy as np
import cv2

import torch
import torch.distributions as td

import alf
from alf.algorithms.data_transformer import ImageScaleTransformer, SimpleDataTransformer
from alf.algorithms.monet_algorithm import MoNetUNet

from vonet.transformer_decoder import DiscreteTransformerDecoder
from vonet.vonet_algorithm import VONetAlgorithm
from vonet.vonet_agent import VONetAgent
from vonet import suite_movi

from alf.environments.gym_wrappers import FrameResize
from alf.utils.schedulers import LinearScheduler, StepScheduler, as_scheduler

import os

# The movi datasets are expected to be like
# `$MOVI_ROOT/movi_a/{train,validation}`
DATA_ROOT = os.environ['MOVI_ROOT']


class MOViDataTransformer(SimpleDataTransformer):
    def __init__(self, observation_spec):
        obs_spec = observation_spec
        transformed_observation_spec = {
            'observation': obs_spec['rgb'],
            'segmentation': obs_spec['segmentation']
        }
        super().__init__(
            transformed_observation_spec=transformed_observation_spec)

    def _transform(self, timestep):
        obs = timestep.observation
        return alf.nest.set_field(
            timestep,
            'observation',
            {
                'observation':
                    obs['rgb'],
                'segmentation':
                    obs['segmentation'
                        ]  # only for evaluation; not used by the model
            })


def _make_vonet_encoder(input_tensor_spec,
                         filters,
                         last_filter=64,
                         use_bn=False,
                         last_activation=alf.math.identity,
                         residue_block=False):
    """Convert an input image into a feature map with spatial size downscaled by
    half.
    """
    cnn_layers = []
    channels = input_tensor_spec.shape[0]
    for i, c in enumerate(filters):
        if residue_block:
            cnn_layers.append(
                alf.layers.ResidueBlock(
                    channels, c, kernel_size=3, stride=2 if i == 0 else 1))
        else:
            cnn_layers.append(
                alf.layers.Conv2D(
                    channels,
                    c,
                    kernel_size=5,
                    strides=2 if i == 0 else 1,
                    padding=2,
                    use_bn=use_bn))
        channels = c
    cnn_layers.append(
        alf.layers.Conv2D(
            channels,
            last_filter,
            kernel_size=1,
            strides=1,
            activation=last_activation))
    # [C,H,W] -> [last_filter,H/2,W/2]
    return alf.nn.Sequential(*cnn_layers, input_tensor_spec=input_tensor_spec)


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


debug = define_config("debug", False)
slot_size = define_config("slot_size", 64)
# MOVi-C contains up to 10 objects; MOVi-D/E contains up to 23 objects.
# See https://github.com/google-research/slot-attention-video/blob/main/savi/configs/movi/savi%2B%2B_conditional.py#L62
n_slots = define_config("n_slots", 11)
# This is 'kappa' in the paper
alpha = define_config("alpha", 0.7)
beta = define_config("beta", 1)
test = define_config("test", False)
# --conf_param="_CONFIG._USER.test_video_path=\'$HOME/movi/pickled/movi_a/validation/video_xxxx.pkl\'"
test_video_path = define_config("test_video_path", None)
dataset = define_config("dataset", "movi_a")
batch_size = define_config("batch_size", 24)
batch_length = define_config("batch_length", 3)
num_envs = define_config("num_envs", 16)
gradient_clip = define_config("gradient_clip", 0.1)
lr = define_config("lr", 1e-4)
slots_interaction_iters = define_config("slots_interaction_iters", 2)

batch_size = 4 if debug else batch_size
num_envs = 4 if debug else num_envs

alf.config('make_ddp_performer', find_unused_parameters=True)

alf.config(
    "create_environment",
    env_load_fn=suite_movi.load,
    env_name="MOVi",
    num_parallel_environments=num_envs)

fields = ['rgb', 'segmentation']

alf.config("ImageChannelFirst", fields=fields)

alf.config(
    "suite_movi.load",
    env_args=dict(
        dataset_root=[DATA_ROOT + f'/{dataset}'],
        test=test,
        test_video_path=test_video_path),
    gym_env_wrappers=[
        partial(FrameResize, width=128, height=128, fields=fields,
                interpolation=cv2.INTER_NEAREST)
    ],
    alf_env_wrappers=[])

# !! alf.config() must happen before get_env() for multi-gpu training because
# some training hyperparameters like ``mini_batch_size`` will be accessed and
# divided among processes.
alf.config(
    'TrainerConfig',
    data_transformer_ctor=[
        partial(
            ImageScaleTransformer,
            min=0.,
            fields=[f for f in fields if f != 'segmentation']),
        MOViDataTransformer
    ],
    algorithm_ctor=partial(
        VONetAgent,
        optimizer=alf.optimizers.AdamTF(
            lr=LinearScheduler("iterations", [(0, lr / 10), (5000, lr),
                                              (100000, lr),
                                              (150000, lr / 10)]),
            gradient_clipping=gradient_clip,
            clip_by_global_norm=True)),
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    temporally_independent_train_step=False,
    use_rollout_state=True,
    summarize_first_interval=False,
    num_iterations=15_0000,
    mini_batch_size=batch_size,
    mini_batch_length=batch_length,
    num_updates_per_train_iter=1,
    num_checkpoints=1,
    unroll_length=2,
    summary_interval=1000,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    initial_collect_steps=500 if debug else 50000,
    replay_buffer_length=10000)


def configure_vonet(obs_spec):
    unet_filters = (32, 64, 64, 128, 128)
    if dataset not in ['movi_a', 'movi_b']:
        unet_filters += (128, )
    monet_attention_unet_cls = partial(
        # size/32
        MoNetUNet,
        filters=unet_filters,
        nonskip_fc_layers=(512, ) * 2)

    final_decoder_filters = 3

    decoder_cls = partial(
        DiscreteTransformerDecoder,
        optimizer=alf.optimizers.AdamTF(
            lr=3 * lr,
            gradient_clipping=gradient_clip,
            clip_by_global_norm=True),
        vae_optimizer=alf.optimizers.AdamTF(lr=10 * lr),
        output_img_spec=alf.TensorSpec((final_decoder_filters, ) +
                                       obs_spec.shape[-2:]),
        beta=0.,
        gumbel_tau=LinearScheduler("iterations", [(0, 1.), (10000, 0.1)]))

    alf.config(
        'VONetAlgorithm',
        n_slots=n_slots,
        slot_size=slot_size,
        attention_unet_cls=monet_attention_unet_cls,
        encoder_cls=partial(
            _make_vonet_encoder,
            filters=(64, 64, 64, 64, 64),
            residue_block=True),
        decoder_cls=decoder_cls)

    alf.config(
        'SequentialSlotVAE',
        slots_interaction_iters=slots_interaction_iters,
        alpha=alpha,
        beta=LinearScheduler("iterations", [(0, 0.), (50000, beta)]))


# It's safe to call get_observation_spec at the end of a config file so that
# when the env is created, all configs have been done.
configure_vonet(alf.get_observation_spec()['observation'])
