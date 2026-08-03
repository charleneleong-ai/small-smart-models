"""Unit tests for the task-battery helpers.

The pure parts only — CI installs torch, not lm_eval, so nothing here imports it."""
import pytest

from smart_quant.eval import CAPABILITY_BATTERY, primary_accuracy


class TestPrimaryAccuracy:
    def test_prefers_acc_norm_over_acc(self) -> None:
        assert primary_accuracy(
            {"acc,none": 0.5, "acc_norm,none": 0.62, "acc_norm_stderr,none": 0.01}) == 0.62

    def test_ignores_stderr_keys(self) -> None:
        assert primary_accuracy({"acc,none": 0.51, "acc_stderr,none": 0.03}) == 0.51

    def test_falls_back_to_exact_match_for_generation_tasks(self) -> None:
        assert primary_accuracy({"exact_match,none": 0.41}) == 0.41

    def test_group_aggregate_resolves_through_acc(self) -> None:
        assert primary_accuracy({"acc,none": 0.37, "n": 100}) == 0.37

    def test_rounds_to_four_decimal_places(self) -> None:
        assert primary_accuracy({"acc_norm,none": 0.123456}) == 0.1235

    def test_raises_on_unknown_metric(self) -> None:
        with pytest.raises(KeyError):
            primary_accuracy({"f1,none": 0.5})


class TestCapabilityBattery:
    def test_spans_reasoning_commonsense_math_knowledge(self) -> None:
        assert {"arc_challenge", "hellaswag", "winogrande", "gsm8k", "mmlu"} <= set(
            CAPABILITY_BATTERY)
