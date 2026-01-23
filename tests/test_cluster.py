"""Tests for cluster utilities."""

from __future__ import annotations

import numpy as np

from driftguard.federate.cluster.cluster import Fp, GroupState


def _make_fp(
    rng: np.random.Generator,
    num_samples: int = 30,
    num_labels: int = 4,
    num_layers: int = 4,
    num_experts: int = 3,
) -> Fp:
    """Create a synthetic Fp with normalized soft labels and random gates."""
    out_softs = rng.random((num_samples, num_labels))
    out_softs /= out_softs.sum(axis=1, keepdims=True)
    gate_activations = rng.random((num_samples, num_layers, num_experts))
    print(out_softs.shape, gate_activations.shape)
    print(Fp.build(out_softs, gate_activations, w_size=1.0))
    return Fp.build(out_softs, gate_activations, w_size=1.0)
    


def test_fp_build_shapes() -> None:
    """Ensure Fp.build handles variable sample sizes and shapes."""
    rng = np.random.default_rng(0)
    fp = _make_fp(rng, num_samples=30, num_labels=4, num_layers=4, num_experts=3)

    assert fp.label_gate_norm.shape == (4, 4, 3)
    assert fp.w.shape == (4,)
    assert np.isfinite(fp.label_gate_norm).all()


def test_group_state_cluster_and_align() -> None:
    """Cluster twice and verify groups persist and align with stored params."""
    rng = np.random.default_rng(1)
    fps = [_make_fp(rng) for _ in range(30)]

    state = GroupState(thr=1e6)
    state.cluster(fps)

    assert state.groups
    assert sum(group.size for group in state.groups) == 30

    for group in state.groups:
        group.params = [np.array([1.0])]

    state.cluster(fps)

    assert state.groups
    assert len(state.groups[0].params) == 1
    assert np.array_equal(state.groups[0].params[0], np.array([1.0]))

if __name__ == "__main__":
    test_fp_build_shapes()
    test_group_state_cluster_and_align()