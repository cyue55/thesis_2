# Copyright (c) 2024 The Lightning team.
# Apache License Version 2.0
# https://github.com/Lightning-AI/torchmetrics

import logging
import os
import warnings
from functools import lru_cache
from typing import override

import numpy as np
import requests
import torch

from .base import BaseMetric
from .registry import MetricRegistry

try:
    import librosa
    import onnxruntime as ort
    from onnxruntime import InferenceSession
except ImportError:
    librosa, ort = None, None

    class InferenceSession:  # noqa: D101
        def __init__(self, **kwargs):
            ...


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
DNSMOS_DIR = "~/.mbchl/DNSMOS"


@MetricRegistry.register("dnsmos")
class DNSMOSMetric(BaseMetric):
    """Deep Noise Suppression Mean Opinion Score (DNSMOS).

    Proposed in [1] and [2].

    .. [1] C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS: A Non-Intrusive Perceptual
       Objective Speech Quality Metric to Evaluate Noise Suppressors", in Proc. ICASSP,
       2021.
    .. [2] C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A Non-Intrusive
       Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors", in
       Proc. ICASSP, 2022.
    """

    to_numpy = False

    def __init__(self, fs=16000, personalized=False, which="OVRL"):
        if which not in ["P808", "OVRL", "SIG", "BAK"]:
            raise ValueError(f"which must be P808, OVRL, SIG or BAK, got {which}")
        self.fs = fs
        self.personalized = personalized
        self.which = which

    @override
    def compute(self, x, y, lengths):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        out = _deep_noise_suppression_mean_opinion_score(x, self.fs, self.personalized)
        out = dict(zip(["P808", "SIG", "BAK", "OVRL"], out.T))
        return out[self.which]


def _prepare_dnsmos(dnsmos_dir):
    url = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master"
    dnsmos_dir = os.path.expanduser(dnsmos_dir)
    for file in [
        "DNSMOS/DNSMOS/model_v8.onnx",
        "DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "DNSMOS/pDNSMOS/sig_bak_ovr.onnx",
    ]:
        saveto = os.path.join(dnsmos_dir, file[7:])
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        if os.path.exists(saveto):
            try:
                _ = InferenceSession(saveto)
                continue
            except Exception as _:
                os.remove(saveto)
        urlf = f"{url}/{file}"
        logging.info(f"downloading {urlf} to {saveto}")
        myfile = requests.get(urlf)
        with open(saveto, "wb") as f:
            f.write(myfile.content)


@lru_cache
def _load_session(path, device, num_threads=None):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        _prepare_dnsmos(DNSMOS_DIR)
    opts = ort.SessionOptions()
    if num_threads is not None:
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
    if device.type == "cpu":
        infs = InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
    elif "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{"device_id": device.index}, {}]
        infs = InferenceSession(
            path,
            providers=providers,
            provider_options=provider_options,
            sess_options=opts,
        )
    elif "CoreMLExecutionProvider" in ort.get_available_providers():
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{"device_id": device.index}, {}]
        infs = InferenceSession(
            path,
            providers=providers,
            provider_options=provider_options,
            sess_options=opts,
        )
    else:
        infs = InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
    return infs


def _audio_melspec(
    audio,
    n_mels=120,
    frame_size=320,
    hop_length=160,
    sr=16000,
    to_db=True,
):
    shape = audio.shape
    audio = audio.reshape(-1, shape[-1])
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = mel_spec.transpose(0, 2, 1)
    mel_spec = mel_spec.reshape(shape[:-1] + mel_spec.shape[1:])
    if to_db:
        for b in range(mel_spec.shape[0]):
            mel_spec[b, ...] = (librosa.power_to_db(mel_spec[b], ref=np.max) + 40) / 40
    return mel_spec


def _polyfit_val(mos, personalized):
    if personalized:
        p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
        p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
        p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
    else:
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    mos[..., 1] = p_sig(mos[..., 1])
    mos[..., 2] = p_bak(mos[..., 2])
    mos[..., 3] = p_ovr(mos[..., 3])
    return mos


def _deep_noise_suppression_mean_opinion_score(
    preds,
    fs,
    personalized,
    device=None,
    num_threads=None,
):
    if librosa is None or ort is None:
        raise ModuleNotFoundError(
            "DNSMOS metric requires that librosa and onnxruntime are installed. "
            "Install as `pip install librosa onnxruntime-gpu`."
        )
    device = torch.device(device) if device is not None else preds.device
    onnx_sess = _load_session(
        f"{DNSMOS_DIR}/{'p' if personalized else ''}DNSMOS/sig_bak_ovr.onnx",
        device,
        num_threads,
    )
    p808_onnx_sess = _load_session(
        f"{DNSMOS_DIR}/DNSMOS/model_v8.onnx",
        device,
        num_threads,
    )
    desired_fs = SAMPLING_RATE
    if fs != desired_fs:
        audio = librosa.resample(preds.cpu().numpy(), orig_sr=fs, target_sr=desired_fs)
    else:
        audio = preds.cpu().numpy()
    len_samples = int(INPUT_LENGTH * desired_fs)
    while audio.shape[-1] < len_samples:
        audio = np.concatenate([audio, audio], axis=-1)
    num_hops = int(np.floor(audio.shape[-1] / desired_fs) - INPUT_LENGTH) + 1
    moss = []
    hop_len_samples = desired_fs
    for idx in range(num_hops):
        audio_seg = audio[
            ...,
            int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples),
        ]
        if audio_seg.shape[-1] < len_samples:
            continue
        shape = audio_seg.shape
        audio_seg = audio_seg.reshape((-1, shape[-1]))
        input_features = np.array(audio_seg).astype("float32")
        p808_input_features = np.array(
            _audio_melspec(audio=audio_seg[..., :-160])
        ).astype("float32")
        if device.type != "cpu" and (
            "CUDAExecutionProvider" in ort.get_available_providers()
            or "CoreMLExecutionProvider" in ort.get_available_providers()
        ):
            try:
                input_features = ort.OrtValue.ortvalue_from_numpy(
                    input_features, device.type, device.index
                )
                p808_input_features = ort.OrtValue.ortvalue_from_numpy(
                    p808_input_features, device.type, device.index
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to use GPU for DNSMOS, reverting to CPU. Error: {e}"
                )
        oi = {"input_1": input_features}
        p808_oi = {"input_1": p808_input_features}
        mos_np = np.concatenate(
            [p808_onnx_sess.run(None, p808_oi)[0], onnx_sess.run(None, oi)[0]],
            axis=-1,
            dtype="float64",
        )
        mos_np = _polyfit_val(mos_np, personalized)
        mos_np = mos_np.reshape(shape[:-1] + (4,))
        moss.append(mos_np)
    return torch.from_numpy(np.mean(np.stack(moss, axis=-1), axis=-1))
