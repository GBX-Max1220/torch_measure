# Copyright (c) 2026 AIMS Foundations. MIT License.

import torch

from torch_measure.metrics.reliability import (
    cronbach_alpha,
    infit_statistics,
    item_total_correlation,
    outfit_statistics,
)


class TestInfitStatistics:
    def test_output_shape(self):
        predicted = torch.full((20, 10), 0.5)
        observed = torch.bernoulli(predicted)
        infit = infit_statistics(predicted, observed)
        assert infit.shape == (10,)

    def test_perfect_fit_near_one(self):
        """When predicted exactly match observed expectations, infit ~ 1.0."""
        torch.manual_seed(42)
        predicted = torch.rand(100, 20)
        observed = torch.bernoulli(predicted)
        infit = infit_statistics(predicted, observed)
        # Should be near 1.0 on average with enough data
        assert infit.mean().item() > 0.5
        assert infit.mean().item() < 2.0

    def test_nonnegative(self):
        predicted = torch.full((20, 10), 0.5)
        observed = torch.bernoulli(predicted)
        infit = infit_statistics(predicted, observed)
        assert (infit >= 0).all()

    def test_missing_response_ignored_with_inferred_or_explicit_mask(self):
        predicted = torch.tensor([[0.8, 0.2], [0.6, 0.3]])
        observed = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
        expected = torch.tensor([1.0, 7.0 / 3.0])
        for mask in (None, ~torch.isnan(observed)):
            actual = infit_statistics(predicted, observed, mask=mask)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_explicit_mask_excludes_finite_response(self):
        predicted = torch.tensor([[0.8], [0.6]])
        observed = torch.tensor([[1.0], [0.0]])
        mask = torch.tensor([[True], [False]])
        actual = infit_statistics(predicted, observed, mask=mask)
        torch.testing.assert_close(actual, torch.tensor([0.25]))


class TestOutfitStatistics:
    def test_output_shape(self):
        predicted = torch.full((20, 10), 0.5)
        observed = torch.bernoulli(predicted)
        outfit = outfit_statistics(predicted, observed)
        assert outfit.shape == (10,)

    def test_nonnegative(self):
        predicted = torch.full((20, 10), 0.5)
        observed = torch.bernoulli(predicted)
        outfit = outfit_statistics(predicted, observed)
        assert (outfit >= 0).all()

    def test_missing_response_ignored_with_inferred_or_explicit_mask(self):
        predicted = torch.tensor([[0.8, 0.2], [0.6, 0.3]])
        observed = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
        expected = torch.tensor([0.875, 7.0 / 3.0])
        for mask in (None, ~torch.isnan(observed)):
            actual = outfit_statistics(predicted, observed, mask=mask)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_explicit_mask_excludes_finite_response(self):
        predicted = torch.tensor([[0.8], [0.6]])
        observed = torch.tensor([[1.0], [0.0]])
        mask = torch.tensor([[True], [False]])
        actual = outfit_statistics(predicted, observed, mask=mask)
        torch.testing.assert_close(actual, torch.tensor([0.25]))


class TestItemTotalCorrelation:
    def test_output_shape(self):
        torch.manual_seed(42)
        data = torch.bernoulli(torch.full((50, 10), 0.5))
        itc = item_total_correlation(data)
        assert itc.shape == (10,)

    def test_range(self):
        torch.manual_seed(42)
        # Use data with structure to get meaningful correlations
        ability = torch.randn(100)
        difficulty = torch.randn(10)
        logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)
        data = torch.bernoulli(torch.sigmoid(logit))
        itc = item_total_correlation(data)
        assert (itc >= -1.01).all()
        assert (itc <= 1.01).all()


class TestCronbachAlpha:
    def test_returns_float(self):
        torch.manual_seed(42)
        data = torch.bernoulli(torch.full((50, 10), 0.5))
        alpha = cronbach_alpha(data)
        assert isinstance(alpha, float)

    def test_structured_data_high_alpha(self):
        """Data with strong unidimensional structure should have high alpha."""
        torch.manual_seed(42)
        ability = torch.randn(100) * 2
        difficulty = torch.randn(20)
        logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)
        data = torch.bernoulli(torch.sigmoid(logit))
        alpha = cronbach_alpha(data)
        assert alpha > 0.5
