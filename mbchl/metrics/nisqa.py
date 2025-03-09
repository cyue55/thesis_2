# Copyright (c) 2021 Gabriel Mittag, Quality and Usability Lab
# MIT License
# https://github.com/gabrielmittag/NISQA

# Copyright (c) 2024 Wangyou Zhang
# Apache License Version 2.0
# https://github.com/urgent-challenge/urgent2024_challenge/

# Copyright (c) 2024 Philippe Gonzalez
# Apache License Version 2.0

import copy
import logging
import math
import os
from functools import lru_cache
from typing import override

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import BaseMetric
from .registry import MetricRegistry

try:
    import librosa
except ImportError:
    librosa = None

NISQA_DIR = "~/.mbchl/NISQA"


@MetricRegistry.register("nisqa")
class NISQAMetric(BaseMetric):
    """Non-Intrusive Speech Quality Assessment (NISQA).

    Proposed in [1] and [2].

    .. [1] G. Mittag and S. Möller, "Non-Intrusive Speech Quality Assessment for
       Super-Wideband Speech Communication Networks", in Proc. ICASSP, 2019.
    .. [2] G. Mittag, B. Naderi, A. Chehadi and S. Möller, "NISQA: A Deep
       CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with
       Crowdsourced Datasets", in Proc. INTERSPEECH, 2021.
    """

    to_numpy = False

    def __init__(self, fs=16000, which="mos_pred"):
        self.fs = fs
        self.which = which

    @override
    def compute(self, x, y, lengths):
        return _non_intrusive_speech_quality_assessment(x, self.fs, self.which)


def _non_intrusive_speech_quality_assessment(x, fs, which="mos_pred"):
    if librosa is None:
        raise ModuleNotFoundError(
            "NISQA metric requires that librosa is installed. Install as "
            "`pip install librosa`."
        )
    model, args = _load_nisqa_model()
    model.eval()
    x = _get_librosa_melspec(x.cpu().numpy(), fs, args)
    x, n_wins = _segment_specs(torch.from_numpy(x), args)
    with torch.no_grad():
        y = model(x, n_wins.expand(x.shape[0]))
    y = dict(zip(["mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred"], y.T))
    return y[which]


@lru_cache
def _load_nisqa_model():
    model_path = os.path.expanduser(os.path.join(NISQA_DIR, "nisqa.tar"))
    if not os.path.exists(model_path):
        _download_weights()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    args = checkpoint["args"]
    model = _NISQADIM(args)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, args


def _download_weights():
    url = (
        "https://github.com/gabrielmittag/NISQA/raw/refs/heads/master/weights/nisqa.tar"
    )
    nisqa_dir = os.path.expanduser(NISQA_DIR)
    os.makedirs(nisqa_dir, exist_ok=True)
    saveto = os.path.join(nisqa_dir, "nisqa.tar")
    if os.path.exists(saveto):
        return
    logging.info(f"downloading {url} to {saveto}")
    myfile = requests.get(url)
    with open(saveto, "wb") as f:
        f.write(myfile.content)


class _NISQADIM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cnn = _Framewise(args)
        self.time_dependency = _TimeDependency(args)
        pool = _Pooling(args)
        self.pool_layers = _get_clones(pool, 5)

    def forward(self, x, n_wins):
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        return torch.cat(out, dim=1)


class _Framewise(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = _AdaptCNN(args)

    def forward(self, x, n_wins):
        x_packed = pack_padded_sequence(
            x, n_wins, batch_first=True, enforce_sorted=False
        )
        x = self.model(x_packed.data.unsqueeze(1))
        x = x_packed._replace(data=x)
        x, _ = pad_packed_sequence(
            x, batch_first=True, padding_value=0.0, total_length=n_wins.max()
        )
        return x


class _AdaptCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pool_1 = args["cnn_pool_1"]
        self.pool_2 = args["cnn_pool_2"]
        self.pool_3 = args["cnn_pool_3"]
        self.dropout = nn.Dropout2d(p=args["cnn_dropout"])
        cnn_pad = (1, 0) if args["cnn_kernel_size"][0] == 1 else (1, 1)
        self.conv1 = nn.Conv2d(
            1, args["cnn_c_out_1"], args["cnn_kernel_size"], padding=cnn_pad
        )
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2 = nn.Conv2d(
            self.conv1.out_channels,
            args["cnn_c_out_2"],
            args["cnn_kernel_size"],
            padding=cnn_pad,
        )
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(
            self.conv2.out_channels,
            args["cnn_c_out_3"],
            args["cnn_kernel_size"],
            padding=cnn_pad,
        )
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(
            self.conv3.out_channels,
            args["cnn_c_out_3"],
            args["cnn_kernel_size"],
            padding=cnn_pad,
        )
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv5 = nn.Conv2d(
            self.conv4.out_channels,
            args["cnn_c_out_3"],
            args["cnn_kernel_size"],
            padding=cnn_pad,
        )
        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)
        self.conv6 = nn.Conv2d(
            self.conv5.out_channels,
            args["cnn_c_out_3"],
            (args["cnn_kernel_size"][0], args["cnn_pool_3"][1]),
            padding=(1, 0),
        )
        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_2))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_3))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.conv6(x)))
        return x.view(-1, self.conv6.out_channels * self.pool_3[0])


class _TimeDependency(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = _SelfAttention(args)

    def forward(self, x, n_wins):
        return self.model(x, n_wins)


class _SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        encoder_layer = _SelfAttentionLayer(args)
        self.norm1 = nn.LayerNorm(args["td_sa_d_model"])
        self.linear = nn.Linear(
            args["cnn_c_out_3"] * args["cnn_pool_3"][0], args["td_sa_d_model"]
        )
        self.layers = _get_clones(encoder_layer, args["td_sa_num_layers"])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, n_wins):
        src = self.linear(src)
        output = src.transpose(1, 0)
        output = self.norm1(output)
        for mod in self.layers:
            output, n_wins = mod(output, n_wins)
        return output.transpose(1, 0), n_wins


class _SelfAttentionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            args["td_sa_d_model"], args["td_sa_nhead"], args["td_sa_dropout"]
        )
        self.linear1 = nn.Linear(args["td_sa_d_model"], args["td_sa_h"])
        self.dropout = nn.Dropout(args["td_sa_dropout"])
        self.linear2 = nn.Linear(args["td_sa_h"], args["td_sa_d_model"])
        self.norm1 = nn.LayerNorm(args["td_sa_d_model"])
        self.norm2 = nn.LayerNorm(args["td_sa_d_model"])
        self.dropout1 = nn.Dropout(args["td_sa_dropout"])
        self.dropout2 = nn.Dropout(args["td_sa_dropout"])
        self.activation = F.relu

    def forward(self, src, n_wins):
        mask = torch.arange(src.shape[0])[None, :] < n_wins[:, None]
        src2 = self.self_attn(src, src, src, key_padding_mask=~mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, n_wins


class _Pooling(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = _PoolAttFF(args)

    def forward(self, x, n_wins):
        return self.model(x, n_wins)


class _PoolAttFF(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear1 = nn.Linear(args["td_sa_d_model"], args["pool_att_h"])
        self.linear2 = nn.Linear(args["pool_att_h"], 1)
        self.linear3 = nn.Linear(args["td_sa_d_model"], 1)
        self.activation = F.relu
        self.dropout = nn.Dropout(args["pool_att_dropout"])

    def forward(self, x, n_wins):
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2, 1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None]
        att[~mask.unsqueeze(1)] = float("-inf")
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)
        return self.linear3(x)


def _get_librosa_melspec(y, sr, args):
    hop_length = int(sr * args["ms_hop_length"])
    win_length = int(sr * args["ms_win_length"])
    # empty mel filter warning is expected when input signal is not fullband
    # see https://github.com/gabrielmittag/NISQA/issues/6#issuecomment-838157571
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=args["ms_n_fft"],
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=args["ms_n_mels"],
        fmin=0.0,
        fmax=args["ms_fmax"],
        htk=False,
        norm="slaney",
    )
    # batch processing of librosa.core.amplitude_to_db is not equivalent to individual
    # processing due to top_db being relative to max value
    # so process individually and then stack
    return np.stack(
        [librosa.amplitude_to_db(m, ref=1.0, amin=1e-4, top_db=80.0) for m in melspec]
    )


def _segment_specs(x, args):
    seg_length = args["ms_seg_length"]
    seg_hop = args["ms_seg_hop_length"]
    max_length = args["ms_max_segments"]
    n_wins = x.shape[2] - (seg_length - 1)
    if n_wins < 1:
        raise RuntimeError("Input signal is too short.")
    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(2, 1)[:, idx3, :].transpose(3, 2)
    x = x[:, ::seg_hop]
    n_wins = math.ceil(n_wins / seg_hop)
    if max_length < n_wins:
        raise RuntimeError(
            "Maximum number of mel spectrogram windows exceeded. Use shorter audio."
        )
    x_padded = torch.zeros((x.shape[0], max_length, x.shape[2], x.shape[3]))
    x_padded[:, :n_wins] = x
    return x_padded, torch.tensor(n_wins)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
