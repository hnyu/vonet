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
from functools import partial

import torch
import torch.nn as nn

import alf
from alf.layers import Conv2D
from alf.data_structures import namedtuple, AlgStep, LossInfo
from alf.algorithms.algorithm import Algorithm
from alf.networks.encoding_networks import ImageDecodingNetworkV2, EncodingNetwork
from alf.utils import tensor_utils, losses
from alf.utils.schedulers import as_scheduler
from vonet.steve_transformer import TransformerDecoder as SteveTransformerDecoder


def _conv_layer_params(patch_size):
    if patch_size == 8:
        return ((64, 4, 4), (64, 1, 1), (64, 1, 1), (64, 1, 1), (128, 2, 2),
                (128, 1, 1), (128, 1, 1), (128, 1, 1))
    else:
        return ((64, 4, 4), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1),
                (64, 1, 1))


def _decoder(patch_size, c, C):
    if patch_size == 8:
        return torch.nn.Sequential(
            Conv2D(c, 128, 1), Conv2D(128, 128, 3, padding=1),
            Conv2D(128, 128, 1), Conv2D(128, 128, 1), Conv2D(
                128, 64 * 4 * 4, 1), torch.nn.PixelShuffle(4),
            Conv2D(64, 64, 3, padding=1), Conv2D(64, 64, 1), Conv2D(64, 64, 1),
            Conv2D(64, 64 * 2 * 2, 1), torch.nn.PixelShuffle(2),
            Conv2D(64, C, 1, alf.math.identity))
    else:
        return torch.nn.Sequential(
            Conv2D(c, 64, 1), Conv2D(64, 64, 3, padding=1), Conv2D(64, 64, 1),
            Conv2D(64, 64, 1), Conv2D(64, 64 * 2 * 2, 1),
            torch.nn.PixelShuffle(2), Conv2D(64, 64, 3, padding=1),
            Conv2D(64, 64, 1), Conv2D(64, 64, 1), Conv2D(64, 64 * 2 * 2, 1),
            torch.nn.PixelShuffle(2), Conv2D(64, C, 1, alf.math.identity))


def _gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    eps = torch.finfo(logits.dtype).tiny
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    y_soft = torch.nn.functional.softmax(gumbels, dim)
    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


class TokenPosEmbedding(torch.nn.Module):
    def __init__(self, L, dim, dropout=0.1):
        super().__init__()
        self._ln = torch.nn.LayerNorm(dim)
        self._dropout = torch.nn.Dropout(dropout)
        self._pos_embedding = torch.nn.Parameter(torch.Tensor(1, L, dim))
        torch.nn.init.trunc_normal_(self._pos_embedding)

    def forward(self, input):
        # input: [B,l,D]
        l = input.shape[1]
        input = input + self._pos_embedding[:, :l, ...]
        return self._ln(self._dropout(input))


TransDecoderInfo = namedtuple(
    "TransDecoderInfo", [
        "loss", "vae", "rec_loss", "per_token_loss", "rec", "pred_rec",
        "per_token_entropy"
    ],
    default_value=())


@alf.configurable
class DiscreteTransformerDecoder(Algorithm):
    def __init__(self,
                 output_img_spec,
                 slot_size,
                 n_slots: int = 1,
                 patch_size: int = 4,
                 vocab_size: int = 4096,
                 d_model: int = 192,
                 transformer_layers: int = 8,
                 transformer_heads: int = 4,
                 gumbel_tau: float = 1.,
                 beta: float = 0.,
                 optimizer: torch.optim.Optimizer = None,
                 vae_optimizer: torch.optim.Optimizer = None,
                 name: str = "TransformerDecoder"):

        super(DiscreteTransformerDecoder, self).__init__(
            optimizer=optimizer, name=name)

        C, H, W = output_img_spec.shape
        assert patch_size in [4, 8]
        assert H % patch_size == 0 and W % patch_size == 0
        h, w = H // patch_size, W // patch_size
        tokens = h * w
        self._tokens = tokens
        self._patch_size = patch_size
        self._vocab_size = vocab_size
        self._h = h
        self._w = w

        conv_layers = _conv_layer_params(patch_size)
        self._patch_encoder = alf.networks.Sequential(
            alf.networks.ImageEncodingNetwork(
                input_channels=C,
                input_size=(H, W),
                conv_layer_params=conv_layers),
            Conv2D(conv_layers[-1][0], vocab_size, 1))
        self._patch_decoder = _decoder(patch_size, vocab_size, C)

        if vae_optimizer is not None:
            self.add_optimizer(vae_optimizer,
                               [self._patch_encoder, self._patch_decoder])

        self._gumbel_tau = as_scheduler(gumbel_tau)
        self._beta = as_scheduler(beta)

        # create the transformer
        self._slots_d_model_layer = torch.nn.Sequential(
            alf.layers.FC(slot_size, 2 * slot_size, torch.relu_),
            alf.layers.FC(2 * slot_size, d_model))
        self._token_embedding = torch.nn.Embedding(vocab_size, d_model)

        self._trans_dec = SteveTransformerDecoder(
            num_blocks=transformer_layers,
            max_len=tokens,
            d_model=d_model,
            num_heads=transformer_heads,
            dropout=0.1)

        self._logits_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            alf.layers.FC(
                input_size=d_model, output_size=vocab_size, use_bias=False))

        # special begin-of-sentence token
        self._bos = torch.nn.Parameter(torch.Tensor(1, 1, d_model))
        torch.nn.init.xavier_uniform_(self._bos)

        self._token_pos_embedding = TokenPosEmbedding(tokens, d_model)

    def _compute_token_logits(self, query, memory):
        """Given a query vec and a memory vec, use transformer to compute a
        logits vec with length equal to the query.
        """
        # query: [B,N,d_model], memory: [B,M,d_model]
        B, N = query.shape[:2]
        M = memory.shape[1]
        query = self._token_pos_embedding(query)
        query = self._trans_dec(query, memory)
        # [B,N,vocab_size]
        logits = self._logits_layer(query)
        return logits

    def _training_token_logits(self, hard_tokens: torch.Tensor,
                               slots: torch.Tensor):
        # hard_tokens: [B,N], slots: [B,G,D]
        B = slots.shape[0]
        # [B,G,d_model]
        memory = self._slots_d_model_layer(slots)
        # [B,N,d_model]
        query = self._token_embedding(hard_tokens)
        # [1,1,d_model] -> [B,1,d_model]
        bos = self._bos.expand(B, -1, -1)
        # [B,1+N,d_model]
        query = torch.cat([bos, query], dim=1)
        return self._compute_token_logits(query[:, :-1, ...], memory)

    @torch.no_grad()
    def autoregressive_decode(self, slots: torch.Tensor):
        """Warning: this function is expensive. Only call it when necessary to
        visualize the reconstructed image.
        """
        # slots: [B,G,D]
        B = slots.shape[0]
        # [B,G,d_model]
        memory = self._slots_d_model_layer(slots)
        # [B,1,d_model]
        query = self._bos.expand(B, -1, -1)
        tokens = []
        for i in range(self._tokens):
            # [B,i+1,vocab_size]
            logits = self._compute_token_logits(query, memory)
            # [B,1]
            new_token = torch.argmax(logits[:, -1:, ...], dim=-1)
            tokens.append(
                torch.nn.functional.one_hot(new_token, self._vocab_size))
            # [B,i+2,d_model]
            query = torch.cat([query, self._token_embedding(new_token)], dim=1)
        # [B,tokens,vocab_size]
        tokens = torch.cat(tokens, dim=1).float()
        img = self._decode(tokens)
        return img

    def _decode(self, tokens):
        # [B,h*w,vocab_size] -> [B,vocab_size,h*w]
        tokens = tokens.transpose(1, 2)
        # [B,vocab_size,h,w]
        tokens = tokens.reshape(*tokens.shape[:2], self._h, self._w)
        # [B,C,H,W]
        img = self._patch_decoder(tokens)
        return img

    def train_step(self, inputs, state=()):
        img, slots = inputs
        # img: [B,C,H,W]
        B, C, H, W = img.shape

        ### Discrete VAE for learning patch tokens ###
        ##############################################
        # [B,vocab_size,h,w]
        logits = self._patch_encoder(img)[0]
        logits = torch.nn.functional.log_softmax(logits, dim=1)
        entropy = -(logits.exp() * logits).sum((1, 2, 3))  # [B]
        soft_tokens = _gumbel_softmax(logits, self._gumbel_tau(), False, dim=1)
        rec = self._patch_decoder(soft_tokens)
        rec_loss = losses.element_wise_squared_loss(img, rec)
        rec_loss = rec_loss.sum((1, 2, 3))

        ### predict tokens from slots ###
        #################################
        # [B,h,w]
        hard_tokens = torch.argmax(soft_tokens, dim=1)
        hard_tokens = hard_tokens.reshape(B, self._tokens)
        # [B,h*w,vocab_size]
        logits = self._training_token_logits(hard_tokens, slots)
        # [B,h*w]
        token_loss = torch.nn.CrossEntropyLoss(reduction='none')(
            logits.transpose(1, 2), hard_tokens)
        token_loss = token_loss.sum(1)

        # [B,h*w,vocab_size]
        with torch.no_grad():
            pred_tokens = torch.nn.functional.one_hot(
                torch.argmax(logits, dim=-1), self._vocab_size).float()
            # [B,vocab_size,h*w]
            pred_tokens = pred_tokens.transpose(1, 2)
            # [B,C,H,W]
            pred_rec = self._patch_decoder(
                pred_tokens.reshape(B, self._vocab_size, self._h, self._w))

        loss = rec_loss + token_loss - self._beta() * entropy
        return AlgStep(
            output=(),
            state=state,
            info=TransDecoderInfo(
                loss=loss,
                per_token_entropy=entropy / self._tokens,
                rec=rec,
                pred_rec=pred_rec,
                rec_loss=rec_loss,
                per_token_loss=token_loss / self._tokens))

    def calc_loss(self, info: TransDecoderInfo):
        with alf.summary.scope(self.name):
            alf.summary.scalar("tau", self._gumbel_tau())
        return LossInfo(
            loss=info.loss,
            extra=info._replace(rec=(), loss=(), vae=(), pred_rec=()))
