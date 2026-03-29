from camillafir.ui.ng_tab_basic import _normalize_filter_type_value


def test_normalize_filter_type_value_accepts_legacy_mixed_label():
    assert _normalize_filter_type_value("Mixed") == "Mixed"
    assert _normalize_filter_type_value("Mixed Phase") == "Mixed"


def test_normalize_filter_type_value_accepts_legacy_asymmetric_variants():
    assert _normalize_filter_type_value("Asymmetric") == "Asymmetric"
    assert _normalize_filter_type_value("Asymmetric (low-latency)") == "Asymmetric"
