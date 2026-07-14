import pytest

from smart_quant.footprint import Footprint, match_tolerance, target_bytes

QWEN_PARAMS = 35_000_000_000
IQ2_M_BYTES = 11_522_702_304  # unsloth UD-IQ2_M.gguf


class TestBpw:
    def test_iq2_m_lands_near_2_6_bpw(self):
        assert Footprint(QWEN_PARAMS, IQ2_M_BYTES).bpw == pytest.approx(2.63, abs=0.05)

    @pytest.mark.parametrize("bpw", [1.75, 2.2, 2.63, 4.0])
    def test_target_bytes_roundtrips_to_bpw(self, bpw):
        b = target_bytes(QWEN_PARAMS, bpw)
        assert Footprint(QWEN_PARAMS, b).bpw == pytest.approx(bpw, abs=1e-6)


class TestMatchTolerance:
    def test_within_3pct_matches(self):
        a = Footprint(QWEN_PARAMS, IQ2_M_BYTES)
        b = Footprint(QWEN_PARAMS, round(IQ2_M_BYTES * 1.02))
        assert match_tolerance(a, b)

    def test_beyond_tolerance_fails(self):
        a = Footprint(QWEN_PARAMS, IQ2_M_BYTES)
        b = Footprint(QWEN_PARAMS, round(IQ2_M_BYTES * 1.10))
        assert not match_tolerance(a, b)

    def test_symmetric_in_argument_order(self):
        a = Footprint(QWEN_PARAMS, IQ2_M_BYTES)
        b = Footprint(QWEN_PARAMS, round(IQ2_M_BYTES * 1.05))
        assert match_tolerance(a, b) == match_tolerance(b, a)
