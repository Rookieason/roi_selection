import numpy as np

from rdith.heatmap_adapter import wrap_existing_heatmap


def test_wrap_existing_tof_doppler_transposes_to_rdith_order():
    energy = np.zeros((2, 3, 4))
    heatmap = wrap_existing_heatmap(
        energy,
        time_s=np.array([0.0, 1.0]),
        doppler_hz=np.arange(4),
        tof_s=np.arange(3),
    )
    assert heatmap.energy.shape == (2, 4, 3)


def test_wrap_existing_aoa_tof_doppler_transposes_to_rdith_order():
    energy = np.zeros((2, 3, 5, 4))
    heatmap = wrap_existing_heatmap(
        energy,
        time_s=np.array([0.0, 1.0]),
        doppler_hz=np.arange(4),
        tof_s=np.arange(3),
        aoa_rad=np.arange(5),
    )
    assert heatmap.energy.shape == (2, 4, 3, 5)
