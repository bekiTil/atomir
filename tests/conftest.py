"""Fixtures for the characterization suite (test doubles live in `doubles.py`)."""

from __future__ import annotations

import pytest

from atomir.stores.json_store import JsonMemoryStore
from doubles import make_service


@pytest.fixture(autouse=True)
def _clear_plan_cache():
    """The plan cache is a module-level singleton; clear it around every test so
    decomposition results never leak between tests."""
    from atomir import atomic_read

    atomic_read._PLAN_CACHE.clear()
    yield
    atomic_read._PLAN_CACHE.clear()


@pytest.fixture
def store(tmp_path):
    return JsonMemoryStore(path=str(tmp_path / "store.json"))


@pytest.fixture
def service(store):
    """Default offline service: stock fakes + json store, production min_sim."""
    return make_service(store)
