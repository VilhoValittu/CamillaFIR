from types import SimpleNamespace

import numpy as np

from camillafir.dsp.dsp_config import CfgReader


def test_cfg_reader_bool_parses_common_string_values():
    cfg = SimpleNamespace(
        t1="1",
        t2="true",
        t3="yes",
        t4="on",
        f1="0",
        f2="false",
        f3="no",
        f4="off",
        f5="",
    )
    reader = CfgReader(cfg)

    assert reader.bool("t1", False) is True
    assert reader.bool("t2", False) is True
    assert reader.bool("t3", False) is True
    assert reader.bool("t4", False) is True
    assert reader.bool("f1", True) is False
    assert reader.bool("f2", True) is False
    assert reader.bool("f3", True) is False
    assert reader.bool("f4", True) is False
    assert reader.bool("f5", True) is False


def test_cfg_reader_bool_invalid_values_fall_back_to_default():
    cfg = SimpleNamespace(a="maybe", b=2, c=np.nan, d=object())
    reader = CfgReader(cfg)

    assert reader.bool("a", True) is True
    assert reader.bool("b", False) is False
    assert reader.bool("c", True) is True
    assert reader.bool("d", False) is False


def test_cfg_reader_float_rejects_nan_and_inf():
    cfg = SimpleNamespace(a="nan", b="inf", c=float("-inf"))
    reader = CfgReader(cfg)

    assert reader.float("a", 1.25) == 1.25
    assert reader.float("b", 2.5) == 2.5
    assert reader.float_allow_zero("c", 0.75) == 0.75


def test_cfg_reader_float_empty_string_falls_back_to_default():
    cfg = SimpleNamespace(a="")
    reader = CfgReader(cfg)

    assert reader.float("a", 3.5) == 3.5
    assert reader.float_allow_zero("a", 0.0) == 0.0


def test_cfg_reader_zero_handling_differs_between_float_variants():
    cfg = SimpleNamespace(a=0.0, b="0")
    reader = CfgReader(cfg)

    assert reader.float("a", 4.0) == 4.0
    assert reader.float_allow_zero("a", 4.0) == 0.0
    assert reader.float("b", 4.0) == 0.0
    assert reader.float_allow_zero("b", 4.0) == 0.0
