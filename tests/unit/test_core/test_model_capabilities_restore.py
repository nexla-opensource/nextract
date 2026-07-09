"""Tests for model capability snapshot/restore isolation used by unit conftest."""

from __future__ import annotations

from nextract.core.model_capabilities import (
    get_model_capabilities,
    register_model_capability,
    reset_model_capabilities,
    restore_model_capabilities,
    snapshot_model_capabilities,
)


def test_restore_model_capabilities_reinstates_snapshot():
    snap = snapshot_model_capabilities()
    assert snap, "builtin capabilities should be present before reset"
    assert get_model_capabilities("gpt-4o").get("vision") is True

    register_model_capability("test-model-caps-restore", {"vision": False})
    assert "test-model-caps-restore" in snapshot_model_capabilities()

    reset_model_capabilities()
    assert snapshot_model_capabilities() == {}
    assert get_model_capabilities("gpt-4o") == {}

    restore_model_capabilities(snap)
    restored = snapshot_model_capabilities()
    assert restored == snap
    assert get_model_capabilities("gpt-4o").get("vision") is True
    assert "test-model-caps-restore" not in restored


class TestUnitConftestRestoresModelCapabilities:
    """Fixture teardown must restore the pre-test snapshot, not only clear.

    Methods run in definition order under default pytest collection so the
    mutation runs before the post-condition check.
    """

    def test_01_clear_model_capabilities(self):
        reset_model_capabilities()
        assert snapshot_model_capabilities() == {}

    def test_02_builtins_restored_by_autouse_fixture(self):
        caps = get_model_capabilities("gpt-4o")
        assert caps.get("vision") is True
        assert snapshot_model_capabilities(), "model capabilities must not stay wiped"
