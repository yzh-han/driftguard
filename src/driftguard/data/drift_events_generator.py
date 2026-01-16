"""Generate drift events for federated learning simulations."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftEvent:
    """Single drift event; duration=0.0 indicates sudden drift.

    Attributes:
        time_step: Time step when drift starts.
        n_aff_client: Number of clients affected.
        duration: Drift duration in steps; 0 means sudden drift.
        drift_dist: Drift distance (integer step count).
    """

    time_step: int
    n_aff_client: int
    duration: int
    drift_dist: int


def generate_drift_events(
    *,
    n_time_steps: int = 100,
    n_clients: int = 10,
    n_sudden: int = 2,
    n_gradual: int = 2,
    n_stage: int = 3,
    aff_client_ratio_range: tuple[float, float] = (0.1, 0.5),
    start: float = 0.0,
    end: float = 1.0,
    dist_range: tuple[int, int] = (1, 3),
    gradual_duration_ratio: float = 0.15,
    seed: int | None = None,
) -> list[DriftEvent]:
    """Generate drift events across staged time windows.

    Each stage spans an equal slice of [start, end] and spawns both sudden and
    gradual events. Sudden events use duration=0.0.

    Args:
        n_time_steps: Total number of time steps in the simulation.
        n_clients: Total number of clients in the simulation.
        n_sudden: Sudden events per stage.
        n_gradual: Gradual events per stage.
        n_stage: Number of stages between start and end.
        aff_client_ratio_range: Min/max ratio of affected clients per event.
        start: Start ratio in [0, 1] (no drift before this).
        end: End ratio in [0, 1] (no drift after this).
        dist_range: Min/max drift distance (steps).
        gradual_duration_ratio: Maximum duration ratio (of n_time_steps).
        seed: Optional random seed.

    Returns:
        List of generated drift events.
    """
    if n_time_steps <= 0:
        raise ValueError("n_time_steps must be positive")
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")
    if n_sudden < 0 or n_gradual < 0:
        raise ValueError("n_sudden and n_gradual must be non-negative")
    if end <= start:
        raise ValueError("end must be greater than start")
    if n_stage <= 0:
        raise ValueError("n_stage must be positive")
    if start < 0.0 or end > 1.0:
        raise ValueError("start/end must satisfy 0.0 <= start < end <= 1.0")
    if (
        aff_client_ratio_range[0] < 0
        or aff_client_ratio_range[1] < 0
        or aff_client_ratio_range[0] > aff_client_ratio_range[1]
        or aff_client_ratio_range[1] > 1.0
    ):
        raise ValueError("aff_client_ratio_range must be within [0, 1] and ordered")
    if dist_range[0] <= 0 or dist_range[1] <= 0 or dist_range[0] > dist_range[1]:
        raise ValueError("dist_range must be positive")
    if gradual_duration_ratio < 0 or gradual_duration_ratio > 1.0:
        raise ValueError("gradual_duration_ratio must be within [0, 1]")

    rng = random.Random(seed)
    events: list[DriftEvent] = []
    stage_length = (end - start) / n_stage
    total_events = n_sudden + n_gradual
    max_duration_steps = max(1, int(gradual_duration_ratio * n_time_steps))

    # Generate events per stage.
    for stage_idx in range(n_stage):
        stage_start_step = int(n_time_steps * (start + stage_idx * stage_length))
        stage_end_step = int(n_time_steps * (start + (stage_idx + 1) * stage_length))
        stage_span = stage_end_step - stage_start_step
        if stage_span <= 0:
            raise ValueError("stage length must be positive")
        if total_events > stage_span:
            raise ValueError("stage is too small for the requested events")

        # Generate drift events.
        time_steps = rng.sample(range(stage_start_step, stage_end_step), k=total_events)

        for idx, t in enumerate(time_steps):
            n_aff_client = rng.randint(
                int(aff_client_ratio_range[0] * n_clients),
                int(aff_client_ratio_range[1] * n_clients),
            )
            drift_dist = rng.randint(dist_range[0], dist_range[1])
            if idx < n_sudden:
                duration = 0
            else:
                max_allowed = min(max_duration_steps, stage_end_step - t)
                duration = rng.randint(1, max_allowed)
            events.append(
                DriftEvent(
                    time_step=t,
                    n_aff_client=n_aff_client,
                    duration=duration,
                    drift_dist=drift_dist,
                )
            )

    return events
