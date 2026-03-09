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
