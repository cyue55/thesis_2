import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import CausalConv2d, CausalGroupNorm, Resample
from .registry import NetRegistry


@NetRegistry.register("diffunet")
class DiffUNet(nn.Module):
    """Union of NCSN++ and ADM modified for speech enhancement.

    NCSN++ was proposed in [1]. ADM was proposed in [2]. NCSN++ was modified for speech
    enhancement in [3].

    .. [1] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon and B. Poole,
       "Score-Based Generative Modeling through Stochastic Differential Equations", in
       Proc. ICLR, 2021.
    .. [2] P. Dhariwal and A. Nichol, "Diffusion Models Beat GANs on Image Synthesis",
       in Proc. NeurIPS, 2021.
    .. [3] J. Richter, S. Welker, J.-M. Lemercier, B. Lay and T. Gerkmann, "Speech
       Enhancement and Dereverberation with Diffusion-Based Generative Models", in
       IEEE/ACM Trans. Audio, Speech, Lang. Process., 2023.

    Parameters
    ----------
    num_freqs : int
        Number of frequency bins in the input.
    in_channels : int
        Number of input channels. Set to 3 for RGB images. For speech enhancement, since
        the input and output are stacked at the input, set to ``2 * (input_channels +
        output_channels)`` for complex spectrograms, or ``(input_channels +
        output_channels)`` for melspectrograms.
    out_channels : int
        Number of output channels. Set to 3 for RGB images. For speech enhancement,
        set to ``2 * output_channels`` for complex spectrograms, or ``output_channels``
        for melspectrograms.
    aux_out_channels : int
        Number of channels in the auxiliary path of the decoder. Ignored if
        ``decoder_type != "skip"``. Set to 3 for RGB images. For speech enhancement,
        set to ``4 * output_channels`` for complex spectrograms, or
        ``2 * output_channels`` for melspectrograms.
    base_channels : int
        Number of input channels in the first layer.
    channel_mult : list[int]
        Channel multiplier for each resolution.
    num_blocks_per_res : int
        Number of U-Net blocks per resolution.
    noise_channel_mult : int
        Channel multiplier for the Fourier feature size.
    emb_channel_mult : int
        Channel multiplier for the noise embedding.
    fir_kernel : list[float]
        Filter coefficients for the resampling layers.
    attn_resolutions : list[int]
        Resolutions at which to use self-attention.
    attn_bottleneck : bool
        Whether to use self-attention in the bottleneck block.
    encoder_type : {"standard", "residual", "skip"}
        Encoder type.
    decoder_type : {"standard", "residual", "skip"}
        Decoder type.
    block_type : {"ncsn", "adm"}
        U-Net block and embedding network type.
    skip_scale : float
        Scale factor for the skip connections.
    dropout : float
        Dropout rate.
    emb_dim : int
        External embedding size.
    causal : bool
        Whether the attention, convolution and normalization layers should be causal.

    Notes
    -----
    Encoder and decoder type description:

    - ``"standard"``: No auxiliary path. The encoder/decoder is a simple stack of
      encoder/decoder blocks. Each encoder/decoder block is a stack of U-Net blocks with
      optional self-attention and down/up-sampling.
    - ``"residual"``: An auxiliary path connects the input of each encoder/decoder block
      to its output. Each auxiliary path has a down/up-sampling layer and a convolution.
      The auxiliary paths are not connected to each other and operate at the same number
      of channels as the main path.
    - ``"skip"``: An auxiliary path is maintained in parallel to the main path all the
      way through. The auxiliary path operates at the same number of channels as the
      input. Channel adaptation convolutions connect the auxiliary path to the main path
      for each resolution at the output of each encoder/decoder block.

    sp-uhh/sgmse uses ``"skip"`` for both the encoder and the decoder, while
    yang-song/score_sde and NVlabs/edm use ``"residual"`` for the encoder and
    ``"standard"`` for the decoder.

    sp-uhh/sgmse adds an output convolution for channel adaptation from 4 to 2. However
    this convolution is applied after the output scaling. Here we apply the convolution
    before instead, such that the raw neural network layers can be isolated from the
    preconditioning as in Karras et al.

    yang-song/score_sde uses a ``Combiner`` module in the skip connections from the
    auxiliary path to the encoder. The Combiner can either concatenate (``"cat"``) or
    sum (``"sum"``) the channels after channel adaptation. However the ``"sum"`` option
    is used in all experiments. Consequently, NVlabs/edm removes the option and
    hard-codes the summation. Similarly, ``combine_method`` is set to ``"sum"`` by
    default in sp-uhh/sgmse. We hard-code the summation here as in NVlabs/edm.

    The dropout rate is set to 0.0 in sp-uhh/sgmse as opposed to 0.1 in
    yang-song/score_sde and NVlabs/edm. This change does not seem to be mentioned in the
    papers.

    """

    def __init__(
        self,
        num_freqs=256,  # same as img_resolution in NVlabs/edm
        in_channels=4,
        out_channels=2,
        aux_out_channels=4,
        base_channels=64,
        channel_mult=[1, 2, 3, 4],
        num_blocks_per_res=1,
        noise_channel_mult=1,
        emb_channel_mult=4,
        fir_kernel=[1, 1],
        attn_resolutions=[],
        attn_bottleneck=True,
        encoder_type="standard",
        decoder_type="standard",
        block_type="adm",
        skip_scale=0.5**0.5,
        dropout=0.0,
        emb_dim=None,
        causal=False,
    ):
        super().__init__()
        assert encoder_type in ["standard", "residual", "skip"]
        assert decoder_type in ["standard", "residual", "skip"]
        assert block_type in ["ncsn", "adm"]

        self.resampler = Resample(fir_kernel, buffer_padding=True, causal=causal)

        noise_channels = base_channels * noise_channel_mult
        emb_channels = base_channels * emb_channel_mult
        self.emb = _EmbeddingBlock(noise_channels, emb_channels, emb_dim, block_type)

        self.input_conv = _Conv2d(in_channels, base_channels, 3, 1, 1, causal=causal)

        num_res = len(channel_mult)
        channels = [base_channels * channel_mult[i] for i in range(num_res)]

        self.encoder = nn.ModuleList(
            _EncoderBlock(
                in_channels=base_channels if i == 0 else channels[i - 1],
                out_channels=channels[i],
                emb_channels=emb_channels,
                block_type=block_type,
                num_blocks=num_blocks_per_res,
                skip_scale=skip_scale,
                dropout=dropout,
                attention=num_freqs >> i in attn_resolutions,
                resampler=None if i == num_res - 1 else self.resampler,
                causal=causal,
            )
            for i in range(num_res)
        )

        if encoder_type != "standard":
            self.aux_downs = nn.ModuleList(
                None
                if i == num_res - 1
                else _AuxiliaryDown(
                    in_channels=(
                        in_channels
                        if encoder_type == "skip" or i == 0
                        else channels[i - 1]
                    ),
                    out_channels=channels[i],
                    resampler=self.resampler,
                    type_=encoder_type,
                    skip_scale=skip_scale,
                    causal=causal,
                )
                for i in range(num_res)
            )
        else:
            self.aux_downs = [None] * num_res

        skip_channels = [base_channels] + [
            channels[i] for i in range(num_res) for _ in self.encoder[i].unet_blocks
        ]

        self.bottleneck_block_1 = _UNetBlock(
            in_channels=channels[-1],
            out_channels=channels[-1],
            emb_channels=emb_channels,
            block_type=block_type,
            skip_scale=skip_scale,
            dropout=dropout,
            causal=causal,
            attention=attn_bottleneck,
        )
        self.bottleneck_block_2 = _UNetBlock(
            in_channels=channels[-1],
            out_channels=channels[-1],
            emb_channels=emb_channels,
            block_type=block_type,
            skip_scale=skip_scale,
            dropout=dropout,
            causal=causal,
        )

        self.decoder = nn.ModuleList(
            _DecoderBlock(
                in_channels=(channels[i] if i == num_res - 1 else channels[i + 1]),
                out_channels=channels[i],
                emb_channels=emb_channels,
                block_type=block_type,
                num_blocks=num_blocks_per_res + 1,
                # + 1 due to the additional skip connection after aux blocks
                skip_scale=skip_scale,
                dropout=dropout,
                attention=num_freqs >> i in attn_resolutions,
                resampler=None if i == num_res - 1 else self.resampler,
                skip_channels=skip_channels,
                causal=causal,
            )
            for i in reversed(range(num_res))
        )

        if decoder_type != "standard":
            self.aux_ups = nn.ModuleList(
                _AuxiliaryUp(
                    in_channels=(
                        channels[i]
                        if decoder_type == "skip" or i == num_res - 1
                        else channels[i + 1]
                    ),
                    out_channels=(
                        aux_out_channels if decoder_type == "skip" else channels[i]
                    ),
                    resampler=None if i == num_res - 1 else self.resampler,
                    type_=decoder_type,
                    causal=causal,
                )
                for i in reversed(range(num_res))
            )
        else:
            self.aux_ups = [None] * num_res

        if decoder_type != "skip":
            # The additional output convolution for channel adaptation from 4 to 2 in
            # sp-uhh/sgmse means that two output convolutions are stacked when
            # decoder_type != "skip", which feels awkward. Since experiments for
            # decoder_type != "skip" are not included in the paper, we take the liberty
            # to use only one output convolution for decoder_type != "skip" here, which
            # feels more natural.
            self.output_conv = nn.Sequential(
                _GroupNorm(channels[0], causal=causal),
                _Conv2d(channels[0], out_channels, 3, 1, 1, causal=causal),
            )
        else:
            self.output_conv = _Conv2d(aux_out_channels, out_channels, 1, causal=causal)

    def forward(self, x, sigma, emb):
        """Forward pass."""
        emb = self.emb(sigma, emb)
        aux = x
        x = self.input_conv(x)
        skips = [x]
        for encoder_block, aux_block in zip(self.encoder, self.aux_downs):
            x, skips = encoder_block(x, emb, skips)
            if aux_block is not None:
                x, aux = aux_block(x, aux)
            skips.append(x)

        x = self.bottleneck_block_1(x, emb)
        x = self.bottleneck_block_2(x, emb)

        aux = None
        for decoder_block, aux_block in zip(self.decoder, self.aux_ups):
            x = decoder_block(x, emb, skips)
            if aux_block is not None:
                x, aux = aux_block(x, aux)
        if aux is None:
            aux = x

        # The output convolution is applied after the output scaling in sp-uhh/sgmse.
        # Here we apply the convolution before instead, such that the raw neural network
        # layers can be isolated from the preconditioning as in Karras et al.
        aux = self.output_conv(aux)
        return aux


class _EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        block_type,
        num_blocks,
        skip_scale,
        dropout,
        attention,
        resampler,
        causal,
    ):
        super().__init__()
        # in encoder blocks self-attention is used in all U-net blocks except the
        # downsampling block
        self.unet_blocks = nn.ModuleList(
            [
                _UNetBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    emb_channels=emb_channels,
                    block_type=block_type,
                    skip_scale=skip_scale,
                    dropout=dropout,
                    causal=causal,
                    attention=False if i == num_blocks else attention,
                    resampler=resampler if i == num_blocks else None,
                    up_or_down="down",
                )
                for i in range(num_blocks if resampler is None else num_blocks + 1)
            ]
        )

    def forward(self, x, emb, skips):
        for i, unet_block in enumerate(self.unet_blocks):
            x = unet_block(x, emb)
            if i != len(self.unet_blocks) - 1:
                skips.append(x)
        return x, skips


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        block_type,
        num_blocks,
        skip_scale,
        dropout,
        attention,
        resampler,
        skip_channels,
        causal,
    ):
        super().__init__()
        # in decoder blocks self-attention is used
        # - in the last U-net block before the upsampling block for NCSN++
        # - in all U-net blocks except the upsampling block for ADM
        self.unet_blocks = nn.ModuleList(
            [
                _UNetBlock(
                    in_channels=(
                        in_channels
                        if i == -1
                        else skip_channels.pop()
                        + (in_channels if i == 0 else out_channels)
                    ),
                    out_channels=in_channels if i == -1 else out_channels,
                    emb_channels=emb_channels,
                    block_type=block_type,
                    skip_scale=skip_scale,
                    dropout=dropout,
                    causal=causal,
                    attention=attention
                    and (block_type == "adm" or i == num_blocks - 1),
                    resampler=resampler if i == -1 else None,
                    up_or_down="up",
                )
                for i in range(0 if resampler is None else -1, num_blocks)
            ]
        )

    def forward(self, x, emb, skips):
        for unet_block in self.unet_blocks:
            if unet_block.resampler is None:
                x = torch.cat([x, skips.pop()], dim=1)
            x = unet_block(x, emb)
        return x


class _AuxiliaryDown(nn.Module):
    def __init__(self, in_channels, out_channels, resampler, type_, skip_scale, causal):
        super().__init__()
        self.resampler = resampler
        self.type_ = type_
        if type_ == "skip":
            self.conv = _Conv2d(in_channels, out_channels, 1, causal=causal)
        else:
            self.conv = _Conv2d(in_channels, out_channels, 3, 1, 1, causal=causal)
        self.skip_scale = skip_scale

    def forward(self, x, aux):
        aux = self.resampler(aux, "down")
        x = x + self.conv(aux)
        if self.type_ == "residual":
            # yang-song/score_sde only scales this skip connection for
            # encoder_type == "residual". An oversight?
            aux = x = x * self.skip_scale
        return x, aux


class _AuxiliaryUp(nn.Module):
    def __init__(self, in_channels, out_channels, resampler, type_, causal):
        super().__init__()
        self.resampler = resampler
        self.type_ = type_
        self.conv = _Conv2d(in_channels, out_channels, 3, 1, 1, causal=causal)
        if type_ == "skip" or resampler is None:
            self.norm = _GroupNorm(in_channels, causal=causal)

    def forward(self, x, aux):
        if self.resampler is not None:
            aux = self.resampler(aux, "up")
        if self.type_ == "skip" or self.resampler is None:
            h = self.conv(F.silu(self.norm(x)))
            aux = h if aux is None else aux + h
        else:
            x = aux = x + self.conv(aux)
        return x, aux


class _UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        block_type,
        skip_scale,
        dropout,
        causal,
        attention=False,
        resampler=None,
        up_or_down="none",
    ):
        # Dropout is set to 0.0 and sp-uhh/sgmse as opposed to 0.1 in
        # yang-song/score_sde and NVlabs/edm. This change does not seem to be mentioned
        # in the papers.
        super().__init__()
        self.skip_scale = skip_scale
        self.norm_1 = _GroupNorm(in_channels, causal=causal)
        self.conv_1 = _Conv2d(in_channels, out_channels, 3, 1, 1, causal=causal)
        self.linear = nn.Linear(
            emb_channels, out_channels * (2 if block_type == "adm" else 1)
        )
        self.norm_2 = _GroupNorm(out_channels, causal=causal)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = _Conv2d(out_channels, out_channels, 3, 1, 1, causal=causal)
        if in_channels != out_channels or (
            block_type == "ncsn" and resampler is not None
        ):
            self.skip_conv = _Conv2d(in_channels, out_channels, 1, causal=causal)
        else:
            self.skip_conv = None
        self.resampler = resampler
        self.up_or_down = up_or_down
        self.attn = _AttentionBlock(out_channels, causal) if attention else None
        self.block_type = block_type

    def forward(self, x, emb):
        h = F.silu(self.norm_1(x))
        if self.resampler is not None:
            h = self.resampler(h, self.up_or_down)
            x = self.resampler(x, self.up_or_down)
        h = self.conv_1(h)
        # the activation is applied to the embedding inside _EmbeddingBlock
        emb = self.linear(emb).unsqueeze(-1).unsqueeze(-1)
        if self.block_type == "adm":
            scale, shift = emb.chunk(2, dim=1)
            h = (scale + 1) * self.norm_2(h) + shift
        else:
            h = self.norm_2(h + emb)
        h = self.conv_2(self.dropout(F.silu(h)))
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        x = self.skip_scale * (x + h)
        if self.attn is not None:
            x = self.skip_scale * self.attn(x)
        return x


class _AttentionBlock(nn.Module):
    def __init__(self, num_channels, causal):
        super().__init__()
        self.norm = _GroupNorm(num_channels, causal=causal)
        self.conv_query = _Conv2d(num_channels, num_channels, 1, causal=causal)
        self.conv_key = _Conv2d(num_channels, num_channels, 1, causal=causal)
        self.conv_value = _Conv2d(num_channels, num_channels, 1, causal=causal)
        self.conv_out = _Conv2d(num_channels, num_channels, 1, causal=causal)
        self.causal = causal

    def forward(self, x):
        # when causal=True, the time dimension is assumed to be the last one
        # TODO: self.norm is not causal!
        N, _, H, W = x.shape

        x_norm = self.norm(x)
        queries = self.conv_query(x_norm)
        keys = self.conv_key(x_norm)
        values = self.conv_value(x_norm)

        queries = queries.reshape(N, -1, H * W).permute(0, 2, 1)
        keys = keys.reshape(N, -1, H * W)
        values = values.reshape(N, -1, H * W).permute(0, 2, 1)

        weights = torch.bmm(queries, keys / keys.shape[1] ** 0.5)

        if self.causal:
            mask = torch.triu(torch.ones(W, W, device=x.device), diagonal=1)
            mask = mask.masked_fill(mask.bool(), float("-inf"))
            mask = mask.repeat((H, H)).reshape(H * W, H * W)
            weights = weights + mask
            # replace NaNs in weights with -inf before softmax to pass causality check
            weights = weights.nan_to_num(float("-inf"))
            # similarly replace NaNs in values with zero
            values = values.nan_to_num(0.0)

        weights = weights.softmax(dim=-1)

        attentions = torch.bmm(weights, values)
        attentions = attentions.permute(0, 2, 1).reshape(N, -1, H, W)
        return x + self.conv_out(attentions)


class _EmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, block_type):
        super().__init__()
        self.fourier_proj = _GaussianFourierProjection(in_channels)
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.linear_2 = nn.Linear(out_channels, out_channels)
        self.linear_emb = nn.Linear(emb_dim, out_channels) if emb_dim else None
        self.block_type = block_type

    def forward(self, x, emb):
        x = x.view(-1)
        x = self.fourier_proj(x)
        if emb is not None and self.block_type != "adm":
            x = x + self.linear_emb(emb)
        x = F.silu(self.linear_1(x))
        x = self.linear_2(x)
        if emb is not None and self.block_type == "adm":
            x = x + self.linear_emb(emb)
        # the second activation is applied here instead of in every _UNetBlock
        return F.silu(x)


class _GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size, scale=16.0):
        super().__init__()
        b = torch.randn(embedding_size // 2) * scale
        self.register_buffer("b", b)

    def forward(self, x):
        x = 2 * math.pi * x.outer(self.b)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return x


class _Conv2d(nn.Module):
    def __init__(self, *args, causal=False, **kwargs):
        super().__init__()
        if causal:
            self.conv = CausalConv2d(*args, **kwargs)
        else:
            self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        return self.conv(x)


class _GroupNorm(nn.Module):
    def __init__(
        self,
        num_channels,
        num_groups=32,
        min_channels_per_group=4,
        eps=1e-6,
        causal=False,
    ):
        super().__init__()
        if causal:
            cls_ = CausalGroupNorm
        else:
            cls_ = nn.GroupNorm
        self.norm = cls_(
            num_groups=min(num_groups, num_channels // min_channels_per_group),
            num_channels=num_channels,
            eps=eps,
        )

    def forward(self, x):
        return self.norm(x)


@NetRegistry.register("ncsnpp")
class NCSNpp(DiffUNet):
    """NCSN++.

    Proposed in [1].

    .. [1] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon and B. Poole,
       "Score-Based Generative Modeling through Stochastic Differential Equations", in
       Proc. ICLR, 2021.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        aux_out_channels=3,
        num_freqs=256,
        base_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_blocks_per_res=4,
        noise_channel_mult=2,
        emb_channel_mult=4,
        fir_kernel=[1, 3, 3, 1],
        attn_resolutions=[16],
        attn_bottleneck=True,
        encoder_type="residual",
        decoder_type="standard",
        block_type="ncsn",
        skip_scale=0.5**0.5,
        dropout=0.0,
        emb_dim=None,
        causal=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            aux_out_channels=aux_out_channels,
            num_freqs=num_freqs,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_blocks_per_res=num_blocks_per_res,
            noise_channel_mult=noise_channel_mult,
            emb_channel_mult=emb_channel_mult,
            fir_kernel=fir_kernel,
            attn_resolutions=attn_resolutions,
            attn_bottleneck=attn_bottleneck,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            block_type=block_type,
            skip_scale=skip_scale,
            dropout=dropout,
            emb_dim=emb_dim,
            causal=causal,
        )


@NetRegistry.register("adm")
class ADM(DiffUNet):
    """ADM.

    Proposed in [1].

    .. [1] P. Dhariwal and A. Nichol, "Diffusion Models Beat GANs on Image Synthesis",
       in Proc. NeurIPS, 2021.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        aux_out_channels=3,
        num_freqs=256,
        base_channels=192,
        channel_mult=[1, 2, 3, 4],
        num_blocks_per_res=3,
        noise_channel_mult=1,
        emb_channel_mult=4,
        fir_kernel=[1, 1],
        attn_resolutions=[32, 16, 8],  # [16, 8] in improved EDM?
        attn_bottleneck=True,
        encoder_type="standard",
        decoder_type="standard",
        block_type="adm",
        skip_scale=1.0,  # 0.5**0.5 in improved EDM?
        dropout=0.0,
        emb_dim=None,
        causal=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            aux_out_channels=aux_out_channels,
            num_freqs=num_freqs,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_blocks_per_res=num_blocks_per_res,
            noise_channel_mult=noise_channel_mult,
            emb_channel_mult=emb_channel_mult,
            fir_kernel=fir_kernel,
            attn_resolutions=attn_resolutions,
            attn_bottleneck=attn_bottleneck,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            block_type=block_type,
            skip_scale=skip_scale,
            dropout=dropout,
            emb_dim=emb_dim,
            causal=causal,
        )
