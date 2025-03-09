import hashlib
import json
import math
import os
import random
import re
import shutil
import warnings
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache

import numpy as np
import soxr
import torch
import yaml


class AttrDict(dict):
    """Dictionary with attribute access support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MathDict(dict):
    """Dictionary with arithmetic operations support."""

    @staticmethod
    def __apply_op(input_, other, op, default):
        if isinstance(other, dict):
            for key, value in other.items():
                input_[key] = op(input_.get(key, default), value)
        elif isinstance(other, (int, float)):
            for key in input_.keys():
                input_[key] = op(input_[key], other)
        return input_

    def __add__(self, other):
        """Addition."""
        return self.__apply_op(MathDict(self), other, lambda x, y: x + y, 0)

    def __sub__(self, other):
        """Subtraction."""
        return self.__apply_op(MathDict(self), other, lambda x, y: x - y, 0)

    def __mul__(self, other):
        """Multiplication."""
        return self.__apply_op(MathDict(self), other, lambda x, y: x * y, 1)

    def __truediv__(self, other):
        """Division."""
        return self.__apply_op(MathDict(self), other, lambda x, y: x / y, 1)

    def __iadd__(self, other):
        """In-place addition."""
        return self.__apply_op(self, other, lambda x, y: x + y, 0)

    def __isub__(self, other):
        """In-place subtraction."""
        return self.__apply_op(self, other, lambda x, y: x - y, 0)

    def __imul__(self, other):
        """In-place multiplication."""
        return self.__apply_op(self, other, lambda x, y: x * y, 1)

    def __itruediv__(self, other):
        """In-place division."""
        return self.__apply_op(self, other, lambda x, y: x / y, 1)

    def __radd__(self, other):
        """Right addition."""
        return self + other

    def __rsub__(self, other):
        """Right subtraction."""
        return self - other

    def __rmul__(self, other):
        """Right multiplication."""
        return self * other

    def __rtruediv__(self, other):
        """Right division."""
        return self / other


class Registry:
    """Registry for classes and functions.

    Helper class to register classes and functions in a registry. To register a class or
    a function to an instance of :class:`Registry`, use the :meth:`register` method as a
    decorator.

    Usage
    -----
    >>> MyRegistry = Registry("registry_name")
    >>> @MyRegistry.register("my_class_key")
    ... class MyClass:
    ...     pass
    >>> my_cls = MyRegistry.get("my_class_key")
    >>> my_obj = my_cls()
    """

    def __init__(self, tag):
        self.tag = tag
        self._registry = {}

    def register(self, key):
        """Register a class or a function."""

        def inner_wrapper(wrapped_class):
            if key in self._registry:
                raise ValueError(f"'{key}' already in {self.tag} registry")
            self._registry[key] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def get(self, key):
        """Get a registered class or function."""
        if key in self._registry:
            return self._registry[key]
        else:
            raise KeyError(f"'{key}' not in {self.tag} registry")

    def keys(self):
        """Get all registered keys."""
        return self._registry.keys()


def impulse(n, idx=0, val=1.0, dtype=torch.float32):
    """Impulse function.

    Equals zero everywhere except at given index.

    Parameters
    ----------
    n : int
        Output length.
    idx : int, optional
        Index of non-zero value.
    val : float, optional
        Non-zero value.
    dtype : torch.dtype, optional
        Output data type.

    Returns
    -------
    torch.Tensor
        Output tensor. Shape ``(n,)``.

    """
    output = torch.zeros(n, dtype=dtype)
    output[idx] = val
    return output


def seed_everything(seed):
    """Seed global random number generators of ``random``, ``numpy`` and ``torch``.

    Parameters
    ----------
    seed : int
        Seed value.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_params(net, requires_grad=True, unique=True):
    """Count trainable parameters in a ``torch.nn.Module``.

    Parameters
    ----------
    net : torch.nn.Module
        Input module.
    requires_grad : bool, optional
        Whether to count trainable parameters only.
    unique : bool, optional
        Whether to count shared parameters only once.

    Returns
    -------
    int
        Number of trainable parameters.

    """
    params = net.parameters()
    if requires_grad:
        params = filter(lambda p: p.requires_grad, params)
    if unique:
        params = {p.data_ptr(): p for p in params}.values()
    return sum(p.numel() for p in params)


def soxr_output_lenght(n, fs_in, fs_out):
    """Calculate the output length of a signal after resampling with `soxr`_.

    Parameters
    ----------
    n : int
        Input length.
    fs_in : int
        Input sampling rate.
    fs_out
        Output sampling rate.

    Returns
    -------
    int
        Output length.


    .. _soxr: https://python-soxr.readthedocs.io/en/stable/

    """
    decimal = Decimal(n * fs_out / fs_in).to_integral_value(rounding=ROUND_HALF_UP)
    return int(decimal)


def pretty_table(d, key_header="", order_by=None, reverse=False, decimals=None):
    """Pretty-print a dictionary of dictionaries as a table.

    The sub-dictionaries must have the same keys. Each row in the table corresponds to
    one entry in the input dictionary.

    Parameters
    ----------
    d : dict
        Dictionary of dictionaries. Sub-dictionaries must have the same keys.
    key_header : str, optional
        Header for the first column.
    order_by : str, optional
        Key by which to order the rows.
    reverse : bool, optional
        Whether to sort in descending order. Ignored if ``order_by`` is ``None``.
    decimals : int, optional
        Number of decimals to round floats to.

    Example
    -------
    >>> dict_ = {
    ...     'id_1': {'name': 'John', 'age': 25},
    ...     'id_2': {'name': 'Jane', 'age': 23},
    ... }
    >>> pretty_table(dict_)
          name  age
    ----  ----  ---
    id_1  John   25
    id_2  Jane   23

    """
    if not d:
        raise ValueError("input is empty")
    # calculate the first column width
    keys = d.keys()
    first_col_width = max(max(len(str(key)) for key in keys), len(key_header))
    # check that all values have the same keys
    values = d.values()
    for i, value in enumerate(values):
        if i == 0:
            sub_keys = value.keys()
        elif value.keys() != sub_keys:
            raise ValueError("values in input do not all have same keys")
    # calculate the width of each column
    col_widths = [first_col_width]
    for key in sub_keys:
        col_width = max(max(len(str(v[key])) for v in values), len(key))
        col_widths.append(col_width)
    # define the row formatting
    row_fmt = " ".join(f"{{:<{width}}} " for width in col_widths)
    # print the header
    lines_to_print = []
    lines_to_print.append(row_fmt.format(key_header, *sub_keys))
    lines_to_print.append(row_fmt.format(*["-" * w for w in col_widths]))
    # order
    if order_by is None:
        iterator = d.items()
    else:
        # type detection
        order_type_cast = float
        for val in d.values():
            try:
                float(val[order_by])
            except ValueError:
                order_type_cast = str
                break
        iterator = sorted(
            ((key, val) for key, val in d.items()),
            key=lambda x: order_type_cast(x[1][order_by]),
            reverse=reverse,
        )
    for key, items in iterator:
        items = [
            f"{x:.{decimals}f}" if isinstance(x, float) and decimals is not None else x
            for x in items.values()
        ]
        row_fmt = " ".join(
            f"{{:>{width}}} " if isinstance(x, (float, int)) else f"{{:>{width}}} "
            for width, x in zip(col_widths, [key, *items])
        )
        lines_to_print.append(row_fmt.format(key, *items))
    # print lines breaking them into groups if longer than console width
    console_width = shutil.get_terminal_size().columns
    first_col_width += 2
    i_width = 1
    while len(lines_to_print[0]) > first_col_width:
        for i, line in enumerate(lines_to_print):
            end, j_width = first_col_width, i_width
            while (
                j_width < len(col_widths)
                and end + col_widths[j_width] + 2 <= console_width
            ):
                end += col_widths[j_width] + 2
                j_width += 1
            print(line[:end])
            lines_to_print[i] = line[:first_col_width] + line[end:]
        i_width = j_width
        print("")


def id_from_dict(d, n=10):
    """Generate a unique ID from a dictionary.

    Parameters
    ----------
    d : dict
        Input dictionary.
    n : int, optional
        Length of the output ID.

    Returns
    -------
    str
        Output ID.

    """

    def sorted_dict(input_dict):
        output_dict = {}
        for key, value in sorted(input_dict.items()):
            if isinstance(value, dict):
                output_dict[key] = sorted_dict(value)
            elif isinstance(value, set):
                output_dict[key] = sorted(value)
            else:
                output_dict[key] = value
        return output_dict

    d = sorted_dict(d)
    s = str(d.items())
    h = hashlib.sha256(s.encode()).hexdigest()
    return h[:n]


def recursive_dict_update(d, u):
    """Recursively update a dictionary in place.

    Parameters
    ----------
    d : dict
        Dictionary to update.
    u : dict
        Dictionary with updates.

    Returns
    -------
    dict
        Updated dictionary.

    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def flatten_dict(d, sep=".", _parent_key=""):
    """Flatten a nested dictionary.

    Parameters
    ----------
    d : dict
        Input dictionary.
    sep : str, optional
        Separator for keys in the output dictionary.

    Returns
    -------
    dict
        Flattened dictionary.

    """
    items = []
    for k, v in d.items():
        new_key = f"{_parent_key}{sep}{k}" if _parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, sep=sep, _parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_yaml(path, return_attr_dict=False):
    """Read a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML file.
    return_attr_dict : bool, optional
        Whether to return an :class:`AttrDict` instance instead of a regular dictionary.

    Returns
    -------
    dict
        Dictionary with the contents of the YAML file.

    """
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.Loader)
    if return_attr_dict:
        return AttrDict(d)
    return d


def dump_yaml(d, path):
    """Dump a dictionary to a YAML file.

    Parameters
    ----------
    d : dict
        Dictionary to dump.
    path : str
        Path to the output YAML file.

    """
    with open(path, "w") as f:
        yaml.dump(d, f)


def read_json(path, return_attr_dict=False):
    """Read a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    return_attr_dict : bool, optional
        Whether to return an :class:`AttrDict` instance instead of a regular dictionary.

    Returns
    -------
    dict
        Dictionary with the contents of the JSON file.

    """
    with open(path) as f:
        d = json.load(f)
    if return_attr_dict:
        return AttrDict(d)
    return d


def dump_json(d, path):
    """Dump a dictionary to a JSON file.

    Parameters
    ----------
    d : dict
        Dictionary to dump.
    path : str
        Path to the output JSON file.

    """
    with open(path, "w") as f:
        json.dump(d, f, indent=4)


def snr(x, y, scale_invariant=False, zero_mean=True, eps=1e-7, lengths=None):
    """Calculate signal-to-noise ratio (SNR) in dB.

    SNR is calculated along the last dimension. If dimensions other than the batch
    dimension are left after reducing the last dimension, these are reduced by taking
    the mean.

    Parameters
    ----------
    x : torch.Tensor
        Input signal. Shape ``(batch, ..., time)``.
    y : torch.Tensor
        Reference signal. Shape ``(batch, ..., time)``.
    scale_invariant : bool, optional
        Whether to calculate scale-invariant SNR.
    zero_mean : bool, optional
        Whether to subtract the mean before calculation.
    eps : float, optional
        Small value to avoid division by zero.
    lengths : torch.Tensor, optional
        Length of signals before batching. Shape ``(batch,)``.

    Returns
    -------
    torch.Tensor
        SNR in dB. Shape ``(batch,)``.

    """
    x, y = apply_mask(x, y, lengths=lengths)
    if zero_mean:
        if lengths is None:
            x = x - x.mean(-1, keepdim=True)
            y = y - y.mean(-1, keepdim=True)
        else:
            x = x - x.sum(-1, keepdim=True) / lengths.view(-1, *[1] * (x.ndim - 1))
            y = y - y.sum(-1, keepdim=True) / lengths.view(-1, *[1] * (x.ndim - 1))
        x, y = apply_mask(x, y, lengths=lengths)
    if scale_invariant:
        alpha = (x * y).sum(-1, keepdim=True) / (y.pow(2).sum(-1, keepdim=True) + eps)
        y = alpha * y
    out = y.pow(2).sum(-1) / ((y - x).pow(2).sum(-1) + eps)
    out = 10 * torch.log10(out + eps)
    if x.ndim > 2:
        out = out.mean(tuple(range(1, x.ndim - 1)))
    return out


def set_snr(speech, noise, snr, zero_mean=True, return_factor=False, _raise=True):
    """Scale noise to achieve a desired signal-to-noise ratio (SNR).

    Parameters
    ----------
    speech : numpy.ndarray
        Speech signal. One-dimensional.
    noise : numpy.ndarray
        Noise signal. One-dimensional.
    snr : float
        Desired SNR in dB.
    zero_mean : bool, optional
        Whether to subtract the mean before calculating SNR. When ``True``, the noise is
        scaled around the mean such that the mean is preserved.
    return_factor : bool, optional
        Whether to return the scaling factor.

    Returns
    -------
    numpy.ndarray
        Scaled noise signal.

    """
    if speech.ndim != 1 or noise.ndim != 1:
        raise ValueError("input signals must be one-dimensional")
    if zero_mean:
        noise_mean = noise.mean()
        noise = noise - noise_mean
        speech = speech - speech.mean()
    else:
        noise_mean = 0.0
    speech_power = np.sum(speech**2)
    noise_power = np.sum(noise**2)
    if noise_power < speech_power * np.finfo(speech_power.dtype).tiny:
        message = "noise power is too small"
        if _raise:
            raise NoiseTooSmallError(message)
        warnings.warn(f"{message}, returning unchanged")
        factor = 1.0
    else:
        factor = 10 ** (-snr / 20) * (speech_power / noise_power) ** 0.5
    noise = noise * factor + noise_mean
    if return_factor:
        return noise, factor
    else:
        return noise


def set_dbfs(x, dbfs, mode="peak", return_factor=False):
    """Scale a signal to achieve a desired dBFS level.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal. One-dimensional.
    dbfs : float
        Desired dBFS level.
    mode : {"peak", "rms", "aes17"}, optional
        Whether to scale relative to peak or RMS value. If ``"aes17"``, ``dbfs`` is
        interpreted as in AES17, i.e. the RMS value of a full-scale sine wave is 0 dBFS.
    return_factor : bool, optional
        Whether to return the scaling factor.

    Returns
    -------
    numpy.ndarray
        Scaled signal.

    """
    if x.ndim != 1:
        raise ValueError("input signal must be one-dimensional")
    if mode == "peak":
        current_level = np.max(np.abs(x))
    elif mode == "rms":
        current_level = np.sqrt(np.mean(x**2))
    elif mode == "aes17":
        current_level = np.sqrt(np.mean(x**2) * 2)
    else:
        raise ValueError(f"invalid mode, got {mode}")
    if current_level == 0.0:
        raise ValueError("input signal is all zeroes")
    factor = 10 ** (dbfs / 20) / current_level
    x = x * factor
    if return_factor:
        return x, factor
    else:
        return x


def apply_mask(*args, lengths=None):
    """Set elements of a tensor after given lengths to zero.

    Parameters
    ----------
    args : torch.Tensor
        Input tensors. Shape ``(batch, ..., time)``.
    lengths : torch.Tensor, optional
        Length of tensors along last axis before batching. Shape ``(batch,)``.

    Returns
    -------
    torch.Tensor
        Output tensors with elements after given lengths set to zero.

    """
    if lengths is None:
        return args
    assert len(lengths) == args[0].shape[0]
    mask = torch.zeros(args[0].shape, device=args[0].device)
    for i, length in enumerate(lengths):
        mask[i, ..., :length] = 1
    return (x * mask for x in args)


def pad(x, n, axis=-1, where="right"):
    """Zero-padding an array along given axis.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    n : int
        Number of zeros to append.
    axis : int, optional
        Axis along which to pad.
    where : {"left", "right", "both"}, optional
        Where to pad the zeros.

    Returns
    -------
    y : numpy.ndarray
        Padded array.

    """
    padding = np.zeros((x.ndim, 2), int)
    if where == "left":
        padding[axis][0] = n
    elif where == "right":
        padding[axis][1] = n
    elif where == "both":
        padding[axis][0] = n
        padding[axis][1] = n
    else:
        raise ValueError(f"where must be left, right or both, got {where}")
    return np.pad(x, padding)


def unfold(x, size, hop, axis=-1, zero_pad=False):
    """Unfold an array into overlapping frames.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    size : int
        Frame size.
    hop : int
        Hop size.
    axis : int, optional
        Axis along which to unfold.
    zero_pad : bool, optional
        Whether to zero-pad the last frame.

    Returns
    -------
    numpy.ndarray
        Framed array. Has one more dimension than the input. Shape ``(..., n_frames)``.

    """
    axis = axis % x.ndim
    x = np.moveaxis(x, axis, -1)
    if zero_pad:
        n_frames = math.ceil(max(x.shape[-1] - size, 0) / hop) + 1
        x = pad(x, (n_frames - 1) * hop + size - x.shape[-1], axis=-1, where="right")
    else:
        n_frames = (x.shape[-1] - size) // hop + 1
    idx = np.arange(size)[None, :] + hop * np.arange(n_frames)[:, None]
    x = x[..., idx]
    return np.moveaxis(x, -1, axis)


def parse_args(args):
    """Parse a list of command-line arguments.

    Parameters
    ----------
    args : list[str]
        List of strings in the form ``key1.key2.<...>.keyN=value``. The type of the
        value is inferred. Supported types are ``int``, ``float``, ``bool``, and
        ``str``.

    Returns
    -------
    dict
        Dictionary with the parsed arguments.

    Example
    -------
    >>> args = [
    ...     "trainer.device=cpu",
    ...     "trainer.use_wandb=False",
    ...     "trainer.train_batch_sampler_kw.batch_size=4",
    ... ]
    >>> parse_args(args)
    {
        "trainer": {
            "device": "cpu",
            "use_wandb": False,
            "train_batch_sampler_kw": {
                "batch_size": 4
            }
        }
    }

    """
    d = {}
    for arg in args:
        keys, value = arg.split("=")
        keys = keys.split(".")
        d_ = d
        for key in keys[:-1]:
            if key not in d_:
                d_[key] = {}
            d_ = d_[key]
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        if value in ["True", "true"]:
            value = True
        elif value in ["False", "false"]:
            value = False
        d_[keys[-1]] = value
    return d


def split_list(a, n):
    """Split a list into ``n`` lists of roughly equal length.

    Parameters
    ----------
    a : list
        Input list.
    n : int
        Number of splits.

    Returns
    -------
    list
        List of ``n`` lists of elements in ``a``.

    """
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def find_files(paths, ext=None, regex=None, blacklist=[], cache=False):
    """Recursively find files in given paths.

    Parameters
    ----------
    paths : str or list[str] or tuple[str, ...]
        Path or list of paths to search in.
    ext : str, optional
        File extension to match.
    regex : str, optional
        Regular expression to match file paths.
    blacklist : list[str], optional
        List of file paths to ignore.
    cache : bool, optional
        Whether to cache the results. If ``True``, the output is a list instead of a
        generator.

    Returns
    -------
    generator[str] or list[str]
        List of file paths.

    """
    # cast to tuple to allow caching
    if isinstance(paths, list):
        paths = tuple(paths)
    elif not isinstance(paths, tuple):
        paths = (paths,)
    if ext is not None:
        ext = "." + ext.lstrip(".")
    if cache:
        return _cached_find_files(paths, ext, regex, tuple(blacklist))
    else:
        return _find_files(paths, ext, regex, blacklist)


def ltas(x, n_fft, hop=None, power=2.0, axis=-1):
    """Calculate the long-term average spectrum of a signal.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal. Shape ``(..., time, ...)``.
    n_fft : int
        FFT frame size.
    hop : int, optional
        Hop size. If ``None``, uses ``n_fft``.
    power : float, optional
        Exponent to apply to the STFT magnitude before averaging over time.
    axis : int, optional
        Time axis of the input signal.

    Returns
    -------
    numpy.ndarray
        Long-term average spectrum. Shape ``(..., n_fft // 2 + 1, ...)``.

    """
    axis = axis % x.ndim
    hop = n_fft if hop is None else hop
    frames = unfold(x, size=n_fft, hop=hop, axis=axis, zero_pad=True)
    mag = np.abs(np.fft.rfft(frames, axis=axis))
    return np.mean(mag**power, axis=-1)


def estimate_bandwidth(
    x, n_fft=512, hop=None, power=2.0, fs=1.0, threshold=-50, eps=1e-10, axis=-1
):
    """Estimate the bandwidth of a signal as in [1].

    The bandwidth is estimated by calculating the long-term average spectrum and finding
    the highest frequency bin that has a level above the peak level minus a threshold.

    The input can be multidimensional. The different channels are not considered
    independent. Therefore, the power spectrum is averaged over all dimensions and the
    output is a single bandwidth value.

    .. [1] E. Bakhturina, V. Lavrukhin, B. Ginsburg and Y. Zhang, "Hi-Fi Multi-Speaker
       English TTS Dataset", in Proc. INTERSPEECH, 2021.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal. Can be multidimensional.
    n_fft : int
        FFT frame size.
    hop : int, optional
        Hop size. If ``None``, uses ``n_fft``.
    power : float, optional
        Exponent to apply to the STFT magnitude before averaging over time.
    fs : float, optional
        Sampling frequency.
    threshold : float, optional
        Threshold in dB. Must be negative.
    eps : float, optional
        Small value to avoid log of 0.
    axis : int, optional
        Time axis of the input signal.

    Returns
    -------
    float
        Estimated bandwidth.

    """
    assert threshold < 0, f"threshold must be negative, got {threshold}"
    x = np.moveaxis(x, axis, -1)  # (..., time)
    x = x.reshape(-1, x.shape[-1])  # (channels, time)
    spec = ltas(x, n_fft, hop, power=power, axis=-1)  # (channels, n_fft // 2 + 1)
    spec = spec.mean(axis=0)  # (n_fft // 2 + 1,)
    spec = 10 * np.log10(spec.clip(min=eps))
    peak = np.max(spec)
    mask = spec > peak + threshold
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    if mask.all():
        return f[-1].item()
    idx = np.argmax(~mask, axis=-1)
    return f[idx].item()


def resample_to_bandwidth(
    x,
    fs=48000,
    available_fs=[8000, 16000, 22050, 24000, 32000, 44100, 48000],
    n_fft=512,
    hop=None,
    power=2.0,
    threshold=-50,
    eps=1e-10,
    axis=-1,
):
    """Downsample a signal after estimating the bandwidth.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal. Can be multidimensional.
    fs : float, optional
        Sampling frequency of input signal.
    available_fs : list[float], optional
        List of available sampling rates to resample to.
    n_fft : int
        FFT frame size used to estimate the bandwidth.
    hop : int, optional
        Hop size used to estimate the bandwidth. If ``None``, uses ``n_fft``.
    power : float, optional
        Exponent to apply to the STFT magnitude when estimating the bandwidth.
    threshold : float, optional
        Threshold in dB used to estimate the bandwidth. Must be negative.
    eps : float, optional
        Small value to avoid log of 0.
    axis : int, optional
        Time axis of the input signal.

    Returns
    -------
    numpy.ndarray
        Resampled signal.

    """
    bandwidth = estimate_bandwidth(
        x,
        n_fft=n_fft,
        hop=hop,
        power=power,
        fs=fs,
        threshold=threshold,
        eps=eps,
        axis=axis,
    )
    # get first available sampling rate above the bandwidth
    available_fs = np.array(sorted(available_fs))
    mask = available_fs >= bandwidth
    if not mask.any():
        raise ValueError("no available sampling rate above estimated bandwidth")
    fs_out_idx = np.argmax(mask)
    fs_out = available_fs[fs_out_idx]
    if fs_out == fs:
        return x
    x = np.moveaxis(x, axis, 0)
    x = soxr.resample(x, fs, fs_out)
    return np.moveaxis(x, 0, axis)


def linear_interpolation(x, xp, fp):
    """Linear interpolation.

    Parameters
    ----------
    x : torch.Tensor
        Points to interpolate. Shape ``(batch_size, m)`` or ``(m,)``.
    xp : torch.Tensor
        Known x-coordinates. Shape ``(batch_size, n)`` or ``(n,)``.
    fp : torch.Tensor
        Known y-coordinates. Shape ``(batch_size, n)`` or ``(n,)``.

    Returns
    -------
    torch.Tensor
        Interpolated y-coordinates. Shape ``(batch_size, m)`` or ``(m,)``.

    """
    assert x.ndim in [1, 2], x.ndim
    assert xp.ndim in [1, 2], xp.ndim
    assert fp.ndim in [1, 2], fp.ndim
    assert xp.shape == fp.shape, (xp.shape, fp.shape)
    squeeze_output = x.ndim == 1 and xp.ndim == 1
    if x.ndim == 1:
        x = x.unsqueeze(0)
        if xp.ndim == 2:
            x = x.expand(xp.shape[0], -1)
    if xp.ndim == 1:
        xp = xp.unsqueeze(0)
        fp = fp.unsqueeze(0)
        if x.shape[0] == 1:
            xp = xp.expand(x.shape[0], -1)
            fp = fp.expand(x.shape[0], -1)
    indices = torch.searchsorted(xp.contiguous(), x.contiguous()) - 1
    indices = indices.clamp(0, xp.shape[-1] - 2)
    xp_left = xp.gather(-1, indices)
    xp_right = xp.gather(-1, indices + 1)
    fp_left = fp.gather(-1, indices)
    fp_right = fp.gather(-1, indices + 1)
    slope = (fp_right - fp_left) / (xp_right - xp_left)
    output = fp_left + slope * (x - xp_left)
    if squeeze_output:
        assert output.shape[0] == 1
        output = output.squeeze(0)
    return output


def random_audiogram(
    generator=None,
    dtype="float32",
    jitter=None,
    batch_size=None,
    tensor=False,
):
    """Generate a random audiogram.

    A random audiogram is selected from the standard audiograms in [1].

    The frequencies are 250, 375, 500, 750, 1000, 1500, 2000, 3000, 4000, and 6000 Hz.

    .. [1] N. Bisgaard, M. S. M. G. Vlaming and M. Dahlquist, "Standard audiograms for
       the IEC 60118-15 measurement procedure", Trends Amplif., 2010.

    Parameters
    ----------
    generator : numpy.random.Generator, optional
        Random number generator. If ``None``, a new generator is created.
    dtype : str, optional
        Data type of the audiogram.
    jitter : float, optional
        Maximum absolute value of jitter to add to the audiogram.
    batch_size : int, optional
        If provided, ``batch_size`` audiograms are generated and stacked along the
        first axis.
    tensor : bool, optional
        If ```True``, a PyTorch tensor is returned instead of a NumPy array.

    Returns
    -------
    numpy.ndarray
        Audiogram. Shape ``(10, 2)`` or ``(batch_size, 10, 2)``, with the first column
        containing frequencies and the second column containing hearing thresholds.

    """
    if batch_size is None:
        if generator is None:
            generator = np.random.default_rng(np.random.randint(0, 2**32))
        standard_audiograms = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NH
            [10, 10, 10, 10, 10, 10, 15, 20, 30, 40],  # N1
            [20, 20, 20, 22.5, 25, 30, 35, 40, 45, 50],  # N2
            [35, 35, 35, 35, 40, 45, 50, 55, 60, 65],  # N3
            [55, 55, 55, 55, 55, 60, 65, 70, 75, 80],  # N4
            [65, 67.5, 70, 72.5, 75, 80, 80, 80, 80, 80],  # N5
            [75, 77.5, 80, 82.5, 85, 90, 90, 95, 100, 100],  # N6
            [90, 92.5, 95, 100, 105, 105, 105, 105, 105, 105],  # N7
            [10, 10, 10, 10, 10, 10, 15, 30, 55, 70],  # S1
            [20, 20, 20, 22.5, 25, 35, 55, 75, 95, 95],  # S2
            [30, 30, 35, 47.5, 60, 70, 75, 80, 80, 85],  # S3
        ]
        audiogram = np.empty((10, 2), dtype=dtype)
        audiogram[:, 0] = [250, 375, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000]
        audiogram[:, 1] = generator.choice(standard_audiograms)
        if jitter is not None:
            audiogram[:, 1] += generator.uniform(-jitter, jitter, size=10)
            audiogram[:, 1] = np.clip(audiogram[:, 1], 0, 105)
    else:
        audiogram = np.stack(
            [
                random_audiogram(
                    generator,
                    dtype,
                    jitter,
                    batch_size=None,
                    tensor=False,
                )
                for _ in range(batch_size)
            ]
        )
    if tensor:
        audiogram = torch.from_numpy(audiogram)
    return audiogram


def nextpow2(x):
    """Smallest power of 2 greater than or equal to input.

    Parameters
    ----------
    x : int or float
        Input number.

    Returns
    -------
    int
        Next power of 2.

    """
    return 2 ** math.ceil(math.log2(x))


@lru_cache
def _cached_find_files(paths, ext, regex, blacklist):
    return list(_find_files(paths, ext, regex, blacklist))


def _find_files(paths, ext, regex, blacklist):
    for path in paths:
        for root, _, files in os.walk(path):
            for file in files:
                fullfile = os.path.join(root, file)
                if (
                    fullfile not in blacklist
                    and (regex is None or re.match(regex, fullfile))
                    and (ext is None or file.endswith(ext))
                ):
                    yield fullfile


class NoiseTooSmallError(Exception):
    """Raised when noise power is too small."""
