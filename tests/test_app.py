"""Integration tests for Streamlit DSAS dashboard using AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_initial_load() -> None:
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "🛡️ Defensive Spectrum-Awareness System"


def test_app_manual_sweep_button() -> None:
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    # Trigger manual sweep button in sidebar
    if len(at.sidebar.button) > 0:
        at.sidebar.button[0].click().run(timeout=30)
        assert not at.exception

