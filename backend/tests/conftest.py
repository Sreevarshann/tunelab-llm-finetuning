import os

import pytest

import tunelab  # noqa: F401  (MLX env)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Isolated data dir + database per test."""
    from tunelab import config, db

    monkeypatch.setenv("TUNELAB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TUNELAB_DISPATCHER", "false")
    config.get_settings.cache_clear()
    db.engine.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()
    db.engine.cache_clear()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("TUNELAB_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test; set TUNELAB_SLOW=1 (Apple Silicon, downloads the pinned base model)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
