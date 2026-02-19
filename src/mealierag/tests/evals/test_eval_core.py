"""
Unit tests for eval_core.
"""

from unittest.mock import MagicMock

import pytest
from eval_core import (
    _calculate_ndcg,
    _expand_ingredient,
    build_ground_truth_filters,
    compute_retrieval_metrics,
    format_ragas_contexts,
    mean,
    parse_expected_properties,
)


class TestExpandIngredient:
    def test_known_category_expands(self):
        result = _expand_ingredient("meat")
        assert "meat" in result
        assert "chicken" in result
        assert "beef" in result

    def test_specific_ingredient_no_expansion(self):
        # "salmon" is not a category key, so no extra expansion
        result = _expand_ingredient("salmon")
        assert result == ["salmon"]

    def test_lowercases_input(self):
        result = _expand_ingredient("MEAT")
        assert result[0] == "meat"
        assert "chicken" in result

    def test_unknown_ingredient_returns_single_item(self):
        result = _expand_ingredient("quinoa")
        assert result == ["quinoa"]


class TestBuildGroundTruthFilters:
    def test_empty_dict_returns_none(self):
        assert build_ground_truth_filters({}) is None

    def test_must_have_ingredients_populates_must(self):
        result = build_ground_truth_filters({"must_have_ingredients": ["chicken"]})
        assert result is not None
        assert result.must is not None
        assert len(result.must) == 1

    def test_must_not_have_ingredients_populates_must_not(self):
        result = build_ground_truth_filters({"must_not_have_ingredients": ["salmon"]})
        assert result is not None
        assert result.must_not is not None
        assert len(result.must_not) >= 1

    def test_meat_category_expands_into_must_not(self):
        result = build_ground_truth_filters({"must_not_have_ingredients": ["meat"]})
        assert result.must_not is not None
        assert len(result.must_not) > 1

    def test_is_healthy_boolean_filter(self):
        result = build_ground_truth_filters({"is_healthy": True})
        assert result is not None
        assert result.must is not None
        assert len(result.must) == 1

    def test_min_rating_range_filter(self):
        result = build_ground_truth_filters({"min_rating": 4.0})
        assert result is not None
        assert result.must is not None

    def test_max_total_time_minutes_range_filter(self):
        result = build_ground_truth_filters({"max_total_time_minutes": 30})
        assert result is not None
        assert result.must is not None

    def test_tools_any_filter(self):
        result = build_ground_truth_filters({"tools": ["oven", "Pan"]})
        assert result is not None
        assert result.must is not None

    def test_tags_is_silently_ignored(self):
        # tags are not yet implemented. filter should be empty (no must or must_not)
        result = build_ground_truth_filters({"tags": ["vegetarian"]})
        assert result is not None
        assert result.must is None
        assert result.must_not is None

    def test_limit_is_silently_ignored(self):
        result = build_ground_truth_filters({"limit": 10})
        assert result is not None
        assert result.must is None
        assert result.must_not is None

    def test_unknown_key_produces_no_filter_conditions(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = build_ground_truth_filters({"unknown_key": "value"})
        assert result is not None
        assert result.must is None
        assert result.must_not is None
        assert "unknown_key" in caplog.text

    def test_combined_must_and_must_not(self):
        result = build_ground_truth_filters(
            {
                "must_have_ingredients": ["chicken"],
                "must_not_have_ingredients": ["salmon"],
                "is_healthy": True,
            }
        )
        assert result is not None
        assert result.must is not None
        assert result.must_not is not None
        assert len(result.must) == 2  # chicken + is_healthy


class TestComputeRetrievalMetrics:
    def test_perfect_retrieval(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        m = compute_retrieval_metrics(retrieved, relevant)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.recall_capped == pytest.approx(1.0)
        assert m.mrr == pytest.approx(1.0)
        assert m.ndcg == pytest.approx(1.0)
        assert m.hit is True
        assert m.relevant_count == 3

    def test_no_relevant_retrieved(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        m = compute_retrieval_metrics(retrieved, relevant)
        assert m.precision == pytest.approx(0.0)
        assert m.recall == pytest.approx(0.0)
        assert m.mrr == pytest.approx(0.0)
        assert m.ndcg == pytest.approx(0.0)
        assert m.hit is False

    def test_empty_retrieved(self):
        m = compute_retrieval_metrics([], {"a", "b"})
        assert m.precision == pytest.approx(0.0)
        assert m.recall == pytest.approx(0.0)
        assert m.hit is False

    def test_empty_relevant_set(self):
        m = compute_retrieval_metrics(["a", "b"], set())
        assert m.precision == pytest.approx(0.0)
        assert m.recall == pytest.approx(0.0)
        assert m.recall_capped == pytest.approx(0.0)
        assert m.relevant_count == 0

    def test_mrr_first_hit_at_rank_2(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        m = compute_retrieval_metrics(retrieved, relevant)
        assert m.mrr == pytest.approx(1 / 2)

    def test_mrr_first_hit_at_rank_3(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        m = compute_retrieval_metrics(retrieved, relevant)
        assert m.mrr == pytest.approx(1 / 3)

    def test_recall_capped_caps_at_k(self):
        # 2 retrieved, 1 relevant-retrieved, but there are 10 relevant total
        # recall_capped = relevant_retrieved / min(k, |relevant|) = 1/2
        retrieved = ["a", "x"]
        relevant = {str(i) for i in range(9)} | {"a"}
        m = compute_retrieval_metrics(retrieved, relevant)
        assert m.recall_capped == pytest.approx(1 / 2)

    def test_partial_retrieval_precision_and_recall(self):
        retrieved = ["a", "b", "x"]
        relevant = {"a", "b", "c", "d"}
        m = compute_retrieval_metrics(retrieved, relevant)
        assert m.precision == pytest.approx(2 / 3)
        assert m.recall == pytest.approx(2 / 4)


class TestCalculateNdcg:
    def test_all_hits_is_one(self):
        result = _calculate_ndcg(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert result == pytest.approx(1.0)

    def test_no_hits_is_zero(self):
        result = _calculate_ndcg(["x", "y"], {"a", "b"}, k=2)
        assert result == pytest.approx(0.0)

    def test_empty_relevant_is_zero(self):
        result = _calculate_ndcg(["a", "b"], set(), k=2)
        assert result == pytest.approx(0.0)

    def test_hit_at_position_2_lower_than_position_1(self):
        # Hit at rank 1 should yield higher nDCG than hit at rank 2
        score_rank1 = _calculate_ndcg(["a", "x"], {"a"}, k=2)
        score_rank2 = _calculate_ndcg(["x", "a"], {"a"}, k=2)
        assert score_rank1 > score_rank2


class TestParseExpectedProperties:
    def test_none_returns_empty_dict(self):
        assert parse_expected_properties(None) == {}

    def test_empty_string_returns_empty_dict(self):
        assert parse_expected_properties("") == {}

    def test_dict_passthrough(self):
        d = {"must_have_ingredients": ["chicken"]}
        assert parse_expected_properties(d) is d

    def test_valid_string_literal(self):
        raw = "{'must_have_ingredients': ['chicken'], 'is_healthy': True}"
        result = parse_expected_properties(raw)
        assert result == {"must_have_ingredients": ["chicken"], "is_healthy": True}

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_expected_properties("!!!not a dict at all!!!")

    def test_string_that_is_not_a_dict_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            parse_expected_properties("[1, 2, 3]")


class TestFormatRagasContexts:
    def _make_hit(
        self, name, rating=4.5, description="", ingredients=None, method=None
    ):
        hit = MagicMock()
        hit.payload = {
            "name": name,
            "rating": rating,
            "description": description,
            "ingredients": ingredients or [],
            "method": method or [],
        }
        return hit

    def test_empty_hits_returns_empty_strings(self):
        context, names = format_ragas_contexts([])
        assert context == ""
        assert names == []

    def test_single_hit_contains_recipe_markers(self):
        hit = self._make_hit("Pasta Carbonara")
        context, names = format_ragas_contexts([hit])
        assert "[RECIPE START]" in context
        assert "[RECIPE END]" in context
        assert "Pasta Carbonara" in context
        assert names == ["Pasta Carbonara"]

    def test_multiple_hits_are_joined(self):
        hits = [self._make_hit("Recipe A"), self._make_hit("Recipe B")]
        context, names = format_ragas_contexts(hits)
        assert "Recipe A" in context
        assert "Recipe B" in context
        assert names == ["Recipe A", "Recipe B"]

    def test_ingredients_are_listed(self):
        hit = self._make_hit("Salad", ingredients=["lettuce", "tomato"])
        context, _ = format_ragas_contexts([hit])
        assert "- lettuce" in context
        assert "- tomato" in context

    def test_missing_name_falls_back_to_unknown(self):
        hit = MagicMock()
        hit.payload = {}
        context, names = format_ragas_contexts([hit])
        assert "Unknown" in context
        assert names == ["Unknown"]


class TestMean:
    def test_normal_case(self):
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty_list_returns_zero(self):
        assert mean([]) == pytest.approx(0.0)

    def test_single_value(self):
        assert mean([5.0]) == pytest.approx(5.0)
