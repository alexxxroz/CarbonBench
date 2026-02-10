import numpy as np
import torch
import pytest

import carbonbench


class TestNormalizedMAE:
	def test_perfect_prediction(self):
		true = np.array([1.0, 2.0, 3.0])
		pred = np.array([1.0, 2.0, 3.0])
		assert carbonbench.normalized_mae(2.0, true, pred) == pytest.approx(0.0)

	def test_known_values(self):
		true = np.array([10.0, 20.0, 30.0])
		pred = np.array([12.0, 18.0, 33.0])
		mean_flux = 20.0
		# errors: |12-10|/20=0.1, |18-20|/20=0.1, |33-30|/20=0.15 → mean=0.1167
		expected = np.mean([2/20, 2/20, 3/20])
		assert carbonbench.normalized_mae(mean_flux, true, pred) == pytest.approx(expected)

	def test_zero_mean_flux(self):
		true = np.array([1.0, 2.0])
		pred = np.array([2.0, 3.0])
		result = carbonbench.normalized_mae(0.0, true, pred)
		assert np.isfinite(result)


class TestRelativeAbsoluteError:
	def test_perfect_prediction(self):
		true = np.array([1.0, 2.0, 3.0])
		pred = np.array([1.0, 2.0, 3.0])
		assert carbonbench.relative_absolute_error(true, pred) == pytest.approx(0.0)

	def test_naive_prediction(self):
		true = np.array([1.0, 2.0, 3.0])
		pred = np.full(3, np.mean(true))
		assert carbonbench.relative_absolute_error(true, pred) == pytest.approx(1.0)

	def test_worse_than_naive(self):
		true = np.array([1.0, 2.0, 3.0])
		pred = np.array([10.0, 20.0, 30.0])
		assert carbonbench.relative_absolute_error(true, pred) > 1.0

	def test_constant_target(self):
		true = np.array([5.0, 5.0, 5.0])
		pred = np.array([6.0, 6.0, 6.0])
		assert carbonbench.relative_absolute_error(true, pred) == np.inf


class TestCustomLoss:
	@pytest.fixture
	def loss_fn(self):
		igbp_w = {"CRO": 1.0, "ENF": 1.0}
		koppen_w = {"C": 1.0, "D": 1.0}
		return carbonbench.CustomLoss(igbp_w, koppen_w, device='cpu')

	def test_zero_loss_on_identical(self, loss_fn):
		preds = torch.ones(2, 5, 3)
		targets = torch.ones(2, 5, 3)
		loss = loss_fn(preds, targets)
		assert loss.item() == pytest.approx(0.0)

	def test_positive_loss_on_different(self, loss_fn):
		preds = torch.zeros(2, 5, 3)
		targets = torch.ones(2, 5, 3)
		loss = loss_fn(preds, targets)
		assert loss.item() > 0

	def test_carbon_balance_constraint(self, loss_fn):
		# 6 output channels triggers the carbon balance branch
		preds = torch.randn(2, 5, 6)
		targets = torch.randn(2, 5, 6)
		loss = loss_fn(preds, targets)
		assert loss.item() > 0

	def test_with_weights(self, loss_fn):
		preds = torch.zeros(2, 5, 3)
		targets = torch.ones(2, 5, 3)
		qc = torch.ones(2, 5)
		# IGBP id 0 = CRO, id 6 = ENF
		igbp = torch.zeros(2, 5, dtype=torch.long)
		koppen = torch.full((2, 5), 2, dtype=torch.long)  # id 2 = C
		loss = loss_fn(preds, targets, qc, igbp, koppen)
		assert loss.item() > 0
