from camillafir.io.auto_mode.filter_priors import (
    get_auto_mode_filter_auto_defaults,
    get_auto_mode_filter_seed_preset,
)


def test_auto_mode_filter_prior_defaults_resolve_linear():
    defaults = get_auto_mode_filter_auto_defaults("Linear Phase")
    assert defaults["filter_type_str"] == "Linear Phase"
    assert defaults["max_boost_db"] == 3.22
    assert defaults["phase_limit"] == 448
    assert defaults["enable_tdc"] is True


def test_auto_mode_filter_prior_seed_resolves_asymmetric_alias():
    seed = get_auto_mode_filter_seed_preset("Asymmetric (low-latency)")
    assert seed["phase_limit"] == 407.2
    assert seed["enable_afdw"] is True
    assert seed["enable_tdc"] is True
    assert seed["filter_smooth"] == 12
