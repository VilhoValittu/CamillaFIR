import numpy as np

from camillafir.io.auto_mode import cache_signature as cs


def test_auto_cache_load_preserves_entries_across_program_version_change(tmp_path, monkeypatch):
    cache_path = tmp_path / "auto_mode_cache.json"
    monkeypatch.setattr(cs, "_auto_cache_path", lambda: str(cache_path))

    cs._auto_cache_save(
        {
            "v": 3,
            "by_filter": {
                "mixed": {
                    "items": {
                        "sig-1": {
                            "best_preset": {"preset_id": "keep-me"},
                            "best_metrics": {"rank_score": 88.2},
                        }
                    },
                    "target_by_measurement": {},
                    "last_used_best": {},
                }
            },
        },
        program_version="1.0.0",
    )

    loaded = cs._auto_cache_load(program_version="2.0.0")
    entry = (
        loaded.get("by_filter", {})
        .get("mixed", {})
        .get("items", {})
        .get("sig-1", {})
    )

    assert entry.get("best_preset", {}).get("preset_id") == "keep-me"
    assert str(loaded.get("program_version", "")) == "1.0.0"


def test_auto_cache_save_can_refresh_program_version_metadata_without_dropping_entries(tmp_path, monkeypatch):
    cache_path = tmp_path / "auto_mode_cache.json"
    monkeypatch.setattr(cs, "_auto_cache_path", lambda: str(cache_path))

    original = cs._auto_cache_empty(program_version="1.0.0")
    original["by_filter"]["mixed"]["items"]["sig-2"] = {
        "best_preset": {"preset_id": "survives-save"},
        "best_metrics": {"rank_score": 91.4},
    }
    cs._auto_cache_save(original, program_version="1.0.0")

    loaded = cs._auto_cache_load(program_version="2.0.0")
    cs._auto_cache_save(loaded, program_version="2.0.0")
    reloaded = cs._auto_cache_load(program_version="2.0.0")

    entry = (
        reloaded.get("by_filter", {})
        .get("mixed", {})
        .get("items", {})
        .get("sig-2", {})
    )

    assert entry.get("best_preset", {}).get("preset_id") == "survives-save"
    assert str(reloaded.get("program_version", "")) == "2.0.0"


def test_auto_signature_changes_when_bass_allpass_state_changes():
    measurements = {
        "f_l": np.asarray([20.0, 80.0], dtype=float),
        "m_l": np.asarray([0.0, -3.0], dtype=float),
        "f_r": np.asarray([20.0, 80.0], dtype=float),
        "m_r": np.asarray([0.0, -3.0], dtype=float),
    }
    base_data = {
        "filter_type": "Asymmetric",
        "bass_integration_allpass_auto_enable": True,
        "bass_integration_allpass_auto_applied": False,
        "bass_integration_allpass_freq_hz": 0.0,
        "bass_integration_allpass_q": 0.707,
    }
    sig_a = cs._auto_signature(
        base_data=base_data,
        measurements=measurements,
        fs_v=48_000,
        taps_v=65_536,
        xos=[],
        hpf=None,
    )
    sig_b = cs._auto_signature(
        base_data={
            **base_data,
            "bass_integration_allpass_auto_applied": True,
            "bass_integration_allpass_freq_hz": 78.0,
            "bass_integration_allpass_q": 0.9,
        },
        measurements=measurements,
        fs_v=48_000,
        taps_v=65_536,
        xos=[],
        hpf=None,
    )

    assert sig_a != sig_b
