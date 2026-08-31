"""Drift monitoring: does it discriminate, and does it refuse to guess?

A drift monitor that reports drift on ordinary traffic is worse than none -- it
teaches whoever reads it to ignore the signal. So the tests that matter here are
symmetric: it must be quiet on in-distribution data AND loud on shifted data.

The first version of this monitored classifier confidence and failed exactly that
way, reporting significant drift on in-distribution messages, because confidence
on a template-generated corpus is bimodal. That is why the classifier services
monitor predicted class mix instead.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.drift import (  # noqa: E402
    MIN_SAMPLES,
    DriftMonitor,
    build_categorical_reference,
    build_reference,
)


def test_identical_distribution_is_stable():
    # The window must hold every observation, or the monitor sees only the tail
    # of the sequence and correctly reports drift -- which is what the first
    # version of this test actually measured.
    reference = build_reference([i / 1000 for i in range(1000)])
    monitor = DriftMonitor(reference, window=1000)
    for value in (i / 1000 for i in range(1000)):
        monitor.observe(value)
    report = monitor.report()
    assert report["status"] == "stable"
    assert report["psi"] < 0.1


def test_shifted_distribution_is_flagged():
    reference = build_reference([i / 1000 for i in range(1000)])
    monitor = DriftMonitor(reference)
    for _ in range(200):
        monitor.observe(0.99)
    report = monitor.report()
    assert report["status"] == "significant_shift"
    assert report["psi"] > 0.25


def test_categorical_mix_is_stable_when_unchanged():
    labels = ["refund"] * 50 + ["complaint"] * 30 + ["tracking"] * 20
    monitor = DriftMonitor(build_categorical_reference(labels))
    for label in labels:
        monitor.observe(label)
    assert monitor.report()["status"] == "stable"


def test_categorical_mix_shift_is_flagged():
    reference = build_categorical_reference(
        ["refund"] * 50 + ["complaint"] * 30 + ["tracking"] * 20
    )
    monitor = DriftMonitor(reference)
    for label in ["tracking"] * 100:
        monitor.observe(label)
    assert monitor.report()["status"] == "significant_shift"


def test_a_class_the_reference_never_saw_is_counted_not_dropped():
    """An unseen class is itself drift; silently ignoring it hides that."""
    monitor = DriftMonitor(build_categorical_reference(["a"] * 60 + ["b"] * 40))
    for _ in range(MIN_SAMPLES + 10):
        monitor.observe("brand_new_class")
    assert monitor.report()["unseen_classes"] == MIN_SAMPLES + 10


def test_a_small_sample_refuses_to_give_a_verdict():
    """PSI on a handful of points is noise; guessing would be worse than waiting."""
    monitor = DriftMonitor(build_reference([i / 100 for i in range(100)]))
    for _ in range(MIN_SAMPLES - 1):
        monitor.observe(0.5)
    assert monitor.report()["status"] == "insufficient_data"


def test_a_missing_reference_does_not_break_the_service():
    """No reference is a reportable state, not an error. The service still serves."""
    monitor = DriftMonitor(None)
    monitor.observe(0.5)
    assert monitor.report()["status"] == "no_reference"


def test_the_window_is_bounded():
    """Unbounded accumulation in a long-running process is a leak."""
    monitor = DriftMonitor(build_reference([i / 100 for i in range(100)]), window=100)
    for value in range(500):
        monitor.observe(value / 500)
    report = monitor.report()
    assert report["observed"] == 100
    assert report["total_seen"] == 500


def test_drift_endpoint_is_served_and_reports_a_status():
    from fastapi.testclient import TestClient

    import api.server as server

    with TestClient(server.app) as client:
        response = client.get("/v1/drift")
        assert response.status_code == 200
        assert "status" in response.json()
