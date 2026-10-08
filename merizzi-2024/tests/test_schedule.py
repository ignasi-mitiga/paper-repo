import math

import pytest
import torch

from merizzi2024.model.schedule import NoiseSchedule


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_schedule_endpoints_and_variance_preservation(kind):
    schedule = NoiseSchedule(kind, min_signal_rate=0.015, max_signal_rate=0.95)
    t = torch.linspace(0, 1, 11)
    noise, signal = schedule(t)
    assert torch.allclose(noise**2 + signal**2, torch.ones_like(t), atol=1e-6)
    assert math.isclose(float(signal[0]), 0.95, abs_tol=1e-4)
    assert math.isclose(float(signal[-1]), 0.015, abs_tol=1e-4)  # float32 sqrt near 1
    assert torch.all(noise[1:] > noise[:-1])  # monotone


def test_cosine_matches_keras_formula():
    schedule = NoiseSchedule("cosine", 0.015, 0.95)
    t = torch.tensor([0.3])
    angle = math.acos(0.95) + 0.3 * (math.acos(0.015) - math.acos(0.95))
    noise, signal = schedule(t)
    assert math.isclose(float(noise), math.sin(angle), abs_tol=1e-6)
    assert math.isclose(float(signal), math.cos(angle), abs_tol=1e-6)


def test_linear_is_linear_in_noise_rate():
    schedule = NoiseSchedule("linear", 0.015, 0.95)
    noise, _ = schedule(torch.tensor([0.0, 0.5, 1.0]))
    assert math.isclose(float(noise[1]), float(noise[0] + noise[2]) / 2, abs_tol=1e-6)


def test_bad_kind_rejected():
    with pytest.raises(ValueError):
        NoiseSchedule("quadratic")
