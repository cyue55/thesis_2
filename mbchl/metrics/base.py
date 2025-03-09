import numpy as np
import torch


class BaseMetric:
    """Base metric class.

    Performs input validation and reshaping for arbitrary input shape support.
    Subclasses must implement the :meth:`compute` method.
    """

    to_numpy = True

    def _check_and_reshape(self, x, y, axis=-1, lengths=None):
        if self.to_numpy:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().float().numpy()
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().float().numpy()
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.cpu().numpy()

        if (
            not isinstance(x, (np.ndarray, torch.Tensor))
            or not isinstance(y, (np.ndarray, torch.Tensor))
            or type(x) is not type(y)
        ):
            raise TypeError(
                f"x and y must be both np.ndarray or torch.Tensor, "
                f"got {type(x)} and {type(y)}"
            )

        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x.shape} and {y.shape}"
            )

        if not isinstance(axis, int):
            raise TypeError(f"axis must be int, got {type(axis)}")

        if -x.ndim <= axis < 0:
            axis = x.ndim + axis
        elif not 0 <= axis < x.ndim:
            raise ValueError(f"axis must be in [-{x.ndim}, {x.ndim - 1}], got {axis}")

        if x.ndim == 1:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            axis += 1
            if lengths is None:
                lengths = [x.shape[-1]]
            elif not isinstance(lengths, int):
                raise TypeError(
                    "lengths must be int for one-dimensional inputs, got "
                    f"{type(lengths)}"
                )
            else:
                lengths = [lengths]
        elif lengths is not None and not isinstance(
            lengths, (list, np.ndarray, torch.Tensor)
        ):
            raise TypeError(
                "lengths must be list, np.ndarray or torch.Tensor for "
                f"multi-dimensional inputs, got {type(lengths)}"
            )

        if isinstance(x, np.ndarray):
            x = np.moveaxis(x, axis, -1)
            y = np.moveaxis(y, axis, -1)
        else:
            x = x.moveaxis(axis, -1)
            y = y.moveaxis(axis, -1)

        if lengths is not None:
            if isinstance(x, np.ndarray):
                lengths = np.asarray(lengths)
            else:
                lengths = torch.as_tensor(lengths, device=x.device)
            if lengths.shape != x.shape[:-1]:
                raise ValueError(
                    "the shape of lengths must be the same as the inputs without the "
                    f"`axis` dimension, got {lengths.shape} and {x.shape[:-1]}"
                )

        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
            y = y.reshape(-1, y.shape[-1])

        if lengths is None:
            if isinstance(x, np.ndarray):
                lengths = np.full(x.shape[0], x.shape[-1], dtype=int)
            else:
                lengths = torch.full((x.shape[0],), x.shape[-1], device=x.device)
        else:
            lengths = lengths.flatten()

        return x, y, lengths

    def __call__(self, x, y, axis=-1, lengths=None):
        """Compute the metric.

        Validates and reshapes the inputs before calling the :meth:`compute` method.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor
            Input signal.
        y : numpy.ndarray or torch.Tensor
            Reference signal.
        axis : int, optional
            Axis along which to compute the metric.
        lengths : list or numpy.ndarray or torch.Tensor, optional
            Original input lengths before batching. Must have same shape as ``x`` and
            ``y`` except for the ``axis`` dimension. For example if inputs have shape
            ``(i, j, k)`` and ``axis = -1``, then ``lengths`` must have shape
            ``(i, j)``. If ``None``, the entire signal length is used.

        Returns
        -------
        float or ndarray
            Metric values. Same shape as ``lengths``.

        """
        x_2d, y_2d, lengths_1d = self._check_and_reshape(x, y, axis, lengths)
        output = self.compute(x_2d, y_2d, lengths_1d)
        if x.ndim == 1:
            output = output.item()
        else:
            output = output.reshape(
                [n for i, n in enumerate(x.shape) if i != axis % x.ndim]
            )
        return output

    def compute(self, x, y, lengths):
        """Compute the metric.

        Parameters
        ----------
        x : ndarray
            Input signal. Shape ``(batch_size, n_samples)``.
        y : ndarray
            Reference signal. Shape ``(batch_size, n_samples)``.
        lengths : ndarray
            Original input lengths before batching. Shape ``(batch_size,)``.

        Returns
        -------
        ndarray
            Metric values. Shape ``(batch_size,)``.

        """
        raise NotImplementedError
