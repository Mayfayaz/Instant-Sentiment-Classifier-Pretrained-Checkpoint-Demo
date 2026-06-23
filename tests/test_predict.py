"""
tests/test_predict.py

Lightweight smoke tests. The full inference test downloads the real
checkpoint, so it's skipped automatically in environments without
network access (e.g. some CI runners) — set RUN_NETWORK_TESTS=1 to force it.
"""

import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_project_structure_exists():
    expected = ["predict.py", "download_model.py", "requirements.txt", "src/model_utils.py"]
    for rel_path in expected:
        assert os.path.isfile(os.path.join(PROJECT_ROOT, rel_path)), f"Missing {rel_path}"


def test_model_utils_importable():
    sys.path.insert(0, PROJECT_ROOT)
    from src import model_utils  # noqa: F401

    assert model_utils.PIPELINE_TASK == "sentiment-analysis"


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="Skips real download/inference unless RUN_NETWORK_TESTS=1 is set.",
)
def test_predict_cli_runs_end_to_end():
    result = subprocess.run(
        [sys.executable, "predict.py", "--text", "I love this!"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0
    assert "Result:" in result.stdout
