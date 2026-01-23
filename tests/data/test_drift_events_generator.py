import pytest

from driftguard.data.drift_simulation import DriftEvent, generate_drift_events


def test_generate_drift_events_count_and_bounds() -> None:
    """Verify event count, boundaries, and sudden/gradual discriminator."""
    events = generate_drift_events(
        n_time_steps=100,
        n_clients=10,
        n_sudden=2,
        n_gradual=3,
        n_stage=4,
        seed=7,
    )

    assert len(events) == 4 * (2 + 3)
    assert sum(1 for event in events if event.duration == 0) == 4 * 2

    for event in events:
        assert isinstance(event, DriftEvent)
        assert 0 <= event.time_step < 100
        assert 0 <= len(event.clients) <= 10
        assert event.duration >= 0
        assert event.drift_dist >= 1


def test_generate_drift_events_is_deterministic() -> None:
    """Ensure deterministic output when the same seed is used."""
    events_a = generate_drift_events(seed=123)
    events_b = generate_drift_events(seed=123)
    assert events_a == events_b


def test_generate_drift_events_validates_ranges() -> None:
    """Ensure invalid configuration inputs raise ValueError."""
    with pytest.raises(ValueError, match="end must be greater than start"):
        generate_drift_events(start=1.0, end=1.0)
    with pytest.raises(ValueError, match="start/end must satisfy"):
        generate_drift_events(start=-0.1, end=0.5)
    with pytest.raises(ValueError, match="n_stage must be positive"):
        generate_drift_events(n_stage=0)
    with pytest.raises(ValueError, match="aff_client_ratio_range must be within"):
        generate_drift_events(aff_client_ratio_range=(-0.1, 0.5))
    with pytest.raises(ValueError, match="dist_range must be positive"):
        generate_drift_events(dist_range=(0, 3))
    with pytest.raises(ValueError, match="gradual_duration_ratio must be within"):
        generate_drift_events(gradual_duration_ratio=-0.1)
    with pytest.raises(ValueError, match="stage is too small"):
        generate_drift_events(n_time_steps=4, n_stage=2, n_sudden=2, n_gradual=2)

if __name__ == "__main__":
    events = generate_drift_events(
        n_time_steps=20,
        n_clients=30,
        n_sudden=2,
        n_gradual=2,
        n_stage=2,
        aff_client_ratio_range=(0.1, 0.3),
        start=0.1,
        end=0.8,
        gradual_duration_ratio=0.2,
    )
    for event in events:
        print(event)