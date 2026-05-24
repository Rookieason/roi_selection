from typing import Optional

import numpy as np

from .types import HeatmapTensor


def wrap_existing_heatmap(
    energy: np.ndarray,
    time_s: np.ndarray,
    doppler_hz: np.ndarray,
    tof_s: Optional[np.ndarray] = None,
    aoa_rad: Optional[np.ndarray] = None,
) -> HeatmapTensor:
    """Convert existing heatmap output to RDITH axis order.

    RDITH uses ``(time, doppler, tof, aoa)``. The current generator commonly
    produces ``(time, tof, doppler)`` or ``(time, tof, aoa, doppler)``; those
    layouts are detected from axis lengths and transposed here.
    """
    energy_arr = np.asarray(energy, dtype=float)
    time_arr = np.asarray(time_s, dtype=float)
    doppler_arr = np.asarray(doppler_hz, dtype=float)
    tof_arr = None if tof_s is None else np.asarray(tof_s, dtype=float)
    aoa_arr = None if aoa_rad is None else np.asarray(aoa_rad, dtype=float)

    if energy_arr.ndim < 2:
        raise ValueError("energy must include at least time and doppler axes")
    if energy_arr.shape[0] != time_arr.size:
        raise ValueError("energy first dimension must match time_s")

    energy_arr = _to_rdith_axis_order(energy_arr, doppler_arr, tof_arr, aoa_arr)
    return HeatmapTensor(
        energy=energy_arr,
        time_s=time_arr,
        doppler_hz=doppler_arr,
        tof_s=tof_arr,
        aoa_rad=aoa_arr,
    )


def _to_rdith_axis_order(
    energy: np.ndarray,
    doppler_hz: np.ndarray,
    tof_s: Optional[np.ndarray],
    aoa_rad: Optional[np.ndarray],
) -> np.ndarray:
    if energy.ndim == 2:
        if energy.shape[1] != doppler_hz.size:
            raise ValueError("2D energy must have shape (time, doppler)")
        return energy

    if energy.ndim == 3:
        if tof_s is None:
            if energy.shape[1] != doppler_hz.size:
                raise ValueError("3D energy without tof_s must use doppler as axis 1")
            return energy
        if energy.shape[1] == doppler_hz.size and energy.shape[2] == tof_s.size:
            return energy
        if energy.shape[1] == tof_s.size and energy.shape[2] == doppler_hz.size:
            return np.transpose(energy, (0, 2, 1))
        raise ValueError("3D energy must be (time, doppler, tof) or (time, tof, doppler)")

    if energy.ndim == 4:
        if tof_s is None or aoa_rad is None:
            raise ValueError("4D energy requires tof_s and aoa_rad axes")
        if (
            energy.shape[1] == doppler_hz.size
            and energy.shape[2] == tof_s.size
            and energy.shape[3] == aoa_rad.size
        ):
            return energy
        if (
            energy.shape[1] == tof_s.size
            and energy.shape[2] == aoa_rad.size
            and energy.shape[3] == doppler_hz.size
        ):
            return np.transpose(energy, (0, 3, 1, 2))
        raise ValueError(
            "4D energy must be (time, doppler, tof, aoa) or (time, tof, aoa, doppler)"
        )

    raise ValueError("energy with more than 4 dimensions is not supported by RDITH yet")
