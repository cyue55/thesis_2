# Copyright (c) 2017 Johns Hopkins University (Shinji Watanabe)
# Apache License 2.0
# https://github.com/espnet/espnet

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from .registry import NetRegistry


@NetRegistry.register("tfgridnet")
class TFGridNet(nn.Module):
    """TF-GridNet.

    Proposed in [1] and [2].

    .. [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim and S. Watanabe,
       "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation", in
       IEEE/ACM Trans. Audio, Speech, Lang. Process., 2023.
    .. [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim and S. Watanabe,
       "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker
       Separation", in Proc. ICASSP, 2023.
    """

    def __init__(
        self,
        output_channels=1,
        input_channels=1,
        n_fft=256,
        layers=6,
        lstm_hidden_units=128,
        attn_heads=4,
        attn_approx_qk_dim=512,
        _emb_dim=32,  # internal emb
        _emb_ks=4,
        _emb_hs=4,
        activation="PReLU",
        eps=1e-5,
        emb_dim=None,  # external emb e.g. speaker emb or audiogram
    ):
        super().__init__()

        self.output_channels = output_channels
        self.layers = layers
        n_freqs = n_fft // 2 + 1

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channels, _emb_dim, ks, padding=padding),
            nn.GroupNorm(1, _emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(layers):
            self.blocks.append(
                _GridNetV2Block(
                    _emb_dim,
                    _emb_ks,
                    _emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_heads,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(
            _emb_dim, output_channels * 2, ks, padding=padding
        )

    def forward(self, batch, emb=None):
        """Forward pass."""
        if emb is not None:
            raise NotImplementedError(
                f"Embedding not implemented for {self.__class__.__name__}"
            )

        # batch.shape = [B, M, F, T]
        batch = batch.transpose(2, 3)  # [B, M, T, F]
        batch = torch.cat((batch.real, batch.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        batch = self.conv(batch)  # [B, -1, T, F]

        for i in range(self.layers):
            batch = self.blocks[i](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, output_channels*2, T, F]

        batch = batch.view([n_batch, self.output_channels, 2, n_frames, n_freqs])
        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])
        batch = batch.transpose(2, 3)
        return batch


class _GridNetV2Block(nn.Module):
    def __init__(
        self,
        _emb_dim,
        _emb_ks,
        _emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="PReLU",
        eps=1e-5,
    ):
        super().__init__()

        in_channels = _emb_dim * _emb_ks

        self.intra_norm = nn.LayerNorm(_emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if _emb_ks == _emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, _emb_dim, _emb_ks, stride=_emb_hs
            )

        self.inter_norm = nn.LayerNorm(_emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if _emb_ks == _emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels * 2, _emb_dim, _emb_ks, stride=_emb_hs
            )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert _emb_dim % n_head == 0

        self.attn_conv_Q = nn.Conv2d(_emb_dim, n_head * E, 1)
        self.attn_norm_Q = _AllHeadPReLULayerNormalization4DCF(
            (n_head, E, n_freqs), eps=eps
        )

        self.attn_conv_K = nn.Conv2d(_emb_dim, n_head * E, 1)
        self.attn_norm_K = _AllHeadPReLULayerNormalization4DCF(
            (n_head, E, n_freqs), eps=eps
        )

        self.attn_conv_V = nn.Conv2d(_emb_dim, n_head * _emb_dim // n_head, 1)
        self.attn_norm_V = _AllHeadPReLULayerNormalization4DCF(
            (n_head, _emb_dim // n_head, n_freqs), eps=eps
        )

        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(_emb_dim, _emb_dim, 1),
            getattr(nn, activation)(),
            _LayerNormalization4DCF((_emb_dim, n_freqs), eps=eps),
        )

        self._emb_dim = _emb_dim
        self._emb_ks = _emb_ks
        self._emb_hs = _emb_hs
        self.n_head = n_head

    def forward(self, x):
        B, C, old_T, old_Q = x.shape

        olp = self._emb_ks - self._emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self._emb_ks) / self._emb_hs) * self._emb_hs
            + self._emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self._emb_ks) / self._emb_hs) * self._emb_hs
            + self._emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        if self._emb_ks == self._emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self._emb_ks * C])
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self._emb_ks, 1), stride=(self._emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2)  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        if self._emb_ks == self._emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self._emb_ks * C])
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self._emb_ks, 1), stride=(self._emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]

            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp : olp + old_T, olp : olp + old_Q]
        batch = inter_rnn

        Q = self.attn_norm_Q(self.attn_conv_Q(batch))
        K = self.attn_norm_K(self.attn_conv_K(batch))
        V = self.attn_norm_V(self.attn_conv_V(batch))
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        _emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (_emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        _emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * _emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self.attn_concat_proj(batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class _LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class _AllHeadPReLULayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 3
        H, E, n_freqs = input_dimension
        param_size = [1, H, E, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, 4)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x
