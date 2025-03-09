# Copyright (c) 2023 Tencent AI Lab
# Apache License Version 2.0
# https://github.com/tencent-ailab/FRA-RIR

import numpy as np
import torch
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import Resample


def fram_rir(
    sr=16000,
    num_src=1,
    num_mic=1,
    room_dim_range=((3.0, 3.0, 2.5), (10.0, 10.0, 4.0)),
    t60_range=(0.1, 0.7),
    direct_range=(-6, 50),
    n_image=(1024, 4097),
    a=-2.0,
    b=2.0,
    tau=0.25,
    src_pattern="omni",
    src_orientation_rad=None,
    mic_pattern="omni",
    mic_orientation_rad=None,
    generator=None,
):
    """Fast Random Approximation of Multi-Channel Room Impulse Response.

    Proposed in [1].

    .. [1] Y. Luo and R. Gu, "Fast Random Approximation of Multi-Channel Room Impulse
       Response", in Proc. ICASSP, 2024.
    """
    if generator is None:
        generator = np.random.default_rng(np.random.randint(0, 2**32))
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(generator.integers(0, 2**32).item())
    room_dim = (
        generator.uniform(room_dim_range[0][0], room_dim_range[1][0]),
        generator.uniform(room_dim_range[0][1], room_dim_range[1][1]),
        generator.uniform(room_dim_range[0][2], room_dim_range[1][2]),
    )
    mic_pos = np.array(
        [
            [
                generator.uniform(0.0, room_dim[0]),
                generator.uniform(0.0, room_dim[1]),
                generator.uniform(0.0, room_dim[2]),
            ]
            for _ in range(num_mic)
        ]
    )
    src_pos = np.array(
        [
            [
                generator.uniform(0.0, room_dim[0]),
                generator.uniform(0.0, room_dim[1]),
                generator.uniform(0.0, room_dim[2]),
            ]
            for _ in range(num_src)
        ]
    )
    T60 = generator.uniform(*t60_range)
    image = generator.choice(range(n_image[0], n_image[1]))
    R = torch.tensor(
        1.0 / (2 * (1.0 / room_dim[0] + 1.0 / room_dim[1] + 1.0 / room_dim[2]))
    )
    mic_position = torch.from_numpy(mic_pos)
    src_position = torch.from_numpy(src_pos)
    direct_dist = (
        (mic_position.unsqueeze(1) - src_position.unsqueeze(0)).pow(2).sum(-1) + 1e-3
    ).sqrt()
    nearest_dist, nearest_mic_idx = direct_dist.min(0)
    nearest_mic_position = mic_position[nearest_mic_idx]
    ns = num_mic * num_src
    ratio = 64
    sample_sr = sr * ratio
    velocity = 340.0
    T60 = torch.tensor(T60)
    direct_idx = (
        torch.ceil(direct_dist * sample_sr / velocity)
        .long()
        .view(
            ns,
        )
    )
    rir_length = int(np.ceil(sample_sr * T60))
    resample1 = Resample(sample_sr, sample_sr // int(np.sqrt(ratio)))
    resample2 = Resample(sample_sr // int(np.sqrt(ratio)), sr)
    reflect_coef = (1 - (1 - torch.exp(-0.16 * R / T60)).pow(2)).sqrt()
    dist_range = [
        torch.linspace(1.0, velocity * T60 / nearest_dist[i] - 1, rir_length)
        for i in range(num_src)
    ]
    dist_prob = torch.linspace(0.0, 1.0, rir_length)
    dist_prob /= dist_prob.sum()
    dist_select_idx = dist_prob.multinomial(
        num_samples=int(image * num_src), replacement=True, generator=torch_generator
    ).view(num_src, image)
    dist_nearest_ratio = torch.stack(
        [dist_range[i][dist_select_idx[i]] for i in range(num_src)], 0
    )
    azm = torch.FloatTensor(num_src, image).uniform_(
        -np.pi, np.pi, generator=torch_generator
    )
    ele = torch.FloatTensor(num_src, image).uniform_(
        -np.pi / 2, np.pi / 2, generator=torch_generator
    )
    unit_3d = torch.stack(
        [
            torch.sin(ele) * torch.cos(azm),
            torch.sin(ele) * torch.sin(azm),
            torch.cos(ele),
        ],
        -1,
    )
    image2nearest_dist = nearest_dist.view(-1, 1, 1) * dist_nearest_ratio.unsqueeze(-1)
    image_position = nearest_mic_position.unsqueeze(1) + image2nearest_dist * unit_3d
    dist = (
        (mic_position.view(-1, 1, 1, 3) - image_position.unsqueeze(0)).pow(2).sum(-1)
        + 1e-3
    ).sqrt()
    reflect_max = (torch.log10(velocity * T60) - 3) / torch.log10(reflect_coef)
    reflect_ratio = (dist / (velocity * T60)) * (reflect_max.view(1, -1, 1) - 1) + 1
    reflect_pertub = torch.FloatTensor(num_src, image).uniform_(
        a, b, generator=torch_generator
    ) * dist_nearest_ratio.pow(tau)
    reflect_ratio = torch.maximum(
        reflect_ratio + reflect_pertub.unsqueeze(0), torch.ones(1)
    )
    dist = torch.cat([direct_dist.unsqueeze(2), dist], 2)
    reflect_ratio = torch.cat([torch.zeros(num_mic, num_src, 1), reflect_ratio], 2)
    delta_idx = (
        torch.minimum(
            torch.ceil(dist * sample_sr / velocity), torch.ones(1) * rir_length - 1
        )
        .long()
        .view(ns, -1)
    )
    delta_decay = reflect_coef.pow(reflect_ratio) / dist
    if src_pattern != "omni":
        img_orientation_rad = torch.FloatTensor(num_src, image, 2).uniform_(
            -np.pi, np.pi, generator=torch_generator
        )
        img_cos_theta = torch.cos(img_orientation_rad[..., 0]) * torch.cos(
            img_orientation_rad[..., 1]
        )
        img_orientation_decay = _freq_invariant_decay_func(
            img_cos_theta, pattern=src_pattern
        )
        if src_orientation_rad is None:
            src_orientation_azi = torch.FloatTensor(num_src).uniform_(
                -np.pi, np.pi, generator=torch_generator
            )
            src_orientation_ele = torch.FloatTensor(num_src).uniform_(
                -np.pi, np.pi, generator=torch_generator
            )
            src_orientation_rad = torch.stack(
                [src_orientation_azi, src_orientation_ele], -1
            )
        else:
            src_orientation_rad = torch.from_numpy(src_orientation_rad)
        src_orientation_decay = _freq_invariant_src_decay_func(
            mic_position, src_position, src_orientation_rad, pattern=src_pattern
        )
        delta_decay[:, :, 0] *= src_orientation_decay
        delta_decay[:, :, 1:] *= img_orientation_decay.unsqueeze(0)
    if mic_pattern != "omni":
        if mic_orientation_rad is None:
            mic_orientation_rad = torch.stack(
                [torch.zeros(num_mic), torch.zeros(num_mic)], -1
            )
        else:
            mic_orientation_rad = torch.from_numpy(mic_orientation_rad)
        all_src_img_pos = torch.cat((src_position.unsqueeze(1), image_position), 1)
        mic_orientation_decay = _freq_invariant_mic_decay_func(
            mic_position, all_src_img_pos, mic_orientation_rad, pattern=mic_pattern
        )
        delta_decay *= mic_orientation_decay
    rir = torch.zeros(ns, rir_length)
    delta_decay = delta_decay.view(ns, -1)
    for i in range(ns):
        remainder_idx = delta_idx[i]
        valid_mask = np.ones(len(remainder_idx))
        while np.sum(valid_mask) > 0:
            valid_remainder_idx, unique_remainder_idx = np.unique(
                remainder_idx, return_index=True
            )
            rir[i][valid_remainder_idx] += (
                delta_decay[i][unique_remainder_idx] * valid_mask[unique_remainder_idx]
            )
            valid_mask[unique_remainder_idx] = 0
            remainder_idx[unique_remainder_idx] = 0
    direct_mask = torch.zeros(ns, rir_length).float()
    for i in range(ns):
        i_start = max(direct_idx[i] + sample_sr * direct_range[0] // 1000, 0)
        i_end = max(direct_idx[i] + sample_sr * direct_range[1] // 1000, i_start + 1)
        i_end = min(i_end, rir_length)
        direct_mask[i, i_start:i_end] = 1.0
        assert direct_mask[i].any()
    rir_direct = rir * direct_mask
    all_rir = torch.stack([rir, rir_direct], 1).view(ns * 2, -1)
    rir_downsample = resample1(all_rir)
    rir_hp = highpass_biquad(rir_downsample, sample_sr // int(np.sqrt(ratio)), 80.0)
    rir = resample2(rir_hp).float().view(num_mic, num_src, 2, -1)
    return rir[:, :, 0].data.numpy(), rir[:, :, 1].data.numpy()


def _freq_invariant_src_decay_func(
    mic_pos, src_pos, src_orientation_rad, pattern="cardioid"
):
    orV_src = _calc_cos(src_orientation_rad).unsqueeze(0)
    rcv_to_src_vec = mic_pos.unsqueeze(1) - src_pos.unsqueeze(0)
    cos_theta = (rcv_to_src_vec * orV_src).sum(-1)
    cos_theta /= torch.sqrt(rcv_to_src_vec.pow(2).sum(-1))
    cos_theta /= torch.sqrt(orV_src.pow(2).sum(-1))
    return _freq_invariant_decay_func(cos_theta, pattern)


def _freq_invariant_mic_decay_func(
    mic_pos, img_pos, mic_orientation_rad, pattern="cardioid"
):
    orV_src = _calc_cos(mic_orientation_rad)
    orV_src = orV_src.view(-1, 1, 1, 3)
    img_to_rcv_vec = img_pos.unsqueeze(0) - mic_pos.unsqueeze(1).unsqueeze(1)
    cos_theta = (img_to_rcv_vec * orV_src).sum(-1)
    cos_theta /= torch.sqrt(img_to_rcv_vec.pow(2).sum(-1))
    cos_theta /= torch.sqrt(orV_src.pow(2).sum(-1))
    return _freq_invariant_decay_func(cos_theta, pattern)


def _freq_invariant_decay_func(cos_theta, pattern="cardioid"):
    if pattern == "cardioid":
        return 0.5 + 0.5 * cos_theta
    elif pattern == "omni":
        return torch.ones_like(cos_theta)
    elif pattern == "bidirectional":
        return cos_theta
    elif pattern == "hyper_cardioid":
        return 0.25 + 0.75 * cos_theta
    elif pattern == "sub_cardioid":
        return 0.75 + 0.25 * cos_theta
    elif pattern == "half_omni":
        c = torch.clamp(cos_theta, 0)
        c[c > 0] = 1.0
        return c
    else:
        raise NotImplementedError


def _calc_cos(orientation_rad):
    return torch.stack(
        [
            torch.cos(orientation_rad[..., 0] * torch.sin(orientation_rad[..., 1])),
            torch.sin(orientation_rad[..., 0] * torch.sin(orientation_rad[..., 1])),
            torch.cos(orientation_rad[..., 1]),
        ],
        -1,
    )
