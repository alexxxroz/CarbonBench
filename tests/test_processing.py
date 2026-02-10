import numpy as np
import pandas as pd
import torch
import pytest

from carbonbench.utils.processing import SlidingWindowDataset, SlidingWindowDatasetTAMRL


def make_synthetic_hist(n_sites=2, n_days=60):
	"""Build a minimal hist dict matching what historical_cache produces."""
	np.random.seed(42)
	hist = {}
	for s in range(n_sites):
		site_name = f"SITE_{s}"
		dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
		n_features = 5
		data = {f"feat_{i}": np.random.randn(n_days) for i in range(n_features)}
		data["GPP_NT_VUT_USTAR50"] = np.random.rand(n_days)
		data["RECO_NT_VUT_USTAR50"] = np.random.rand(n_days)
		data["NEE_VUT_USTAR50"] = np.random.rand(n_days)
		data["NEE_VUT_USTAR50_QC"] = np.ones(n_days)
		data["date"] = dates
		data["site"] = site_name
		data["IGBP"] = "ENF"
		data["Koppen"] = "C"
		data["Koppen_short"] = "Cfb"
		hist[site_name] = pd.DataFrame(data)
	return hist


TARGETS = ["GPP_NT_VUT_USTAR50", "RECO_NT_VUT_USTAR50", "NEE_VUT_USTAR50"]
WINDOW = 30
STRIDE = 15


class TestSlidingWindowDataset:
	@pytest.fixture
	def dataset(self):
		hist = make_synthetic_hist()
		return SlidingWindowDataset(
			hist, TARGETS, include_qc=True,
			window_size=WINDOW, stride=STRIDE,
			cat_features=["IGBP", "Koppen", "Koppen_short"]
		)

	def test_len_positive(self, dataset):
		assert len(dataset) > 0

	def test_getitem_returns_six_tensors(self, dataset):
		sample = dataset[0]
		assert len(sample) == 6

	def test_dynamic_shape(self, dataset):
		x, cat, y, qc, igbp, koppen = dataset[0]
		assert x.shape[0] == WINDOW
		assert cat.shape[0] == WINDOW
		assert y.shape[0] == STRIDE
		assert y.shape[1] == len(TARGETS)

	def test_dtypes(self, dataset):
		x, cat, y, qc, igbp, koppen = dataset[0]
		assert x.dtype == torch.float32
		assert cat.dtype == torch.float32
		assert y.dtype == torch.float32
		assert qc.dtype == torch.float32
		assert igbp.dtype == torch.int64
		assert koppen.dtype == torch.int64

	def test_get_sites(self, dataset):
		sites = dataset.get_sites()
		assert set(sites) == {"SITE_0", "SITE_1"}

	def test_site_indices_nonempty(self, dataset):
		for site in dataset.get_sites():
			indices = dataset.get_site_indices(site)
			assert len(indices) > 0

	def test_no_qc(self):
		hist = make_synthetic_hist()
		ds = SlidingWindowDataset(
			hist, TARGETS, include_qc=False,
			window_size=WINDOW, stride=STRIDE,
			cat_features=["IGBP", "Koppen", "Koppen_short"]
		)
		_, _, y, qc, _, _ = ds[0]
		assert y.shape[1] == len(TARGETS)
		assert (qc.numpy() == 1.0).all()


class TestSlidingWindowDatasetTAMRL:
	@pytest.fixture
	def dataset(self):
		hist = make_synthetic_hist()
		return SlidingWindowDatasetTAMRL(
			hist, TARGETS, include_qc=True,
			window_size=WINDOW, stride=STRIDE,
			cat_features=["IGBP", "Koppen", "Koppen_short"]
		)

	def test_getitem_returns_eight_tensors(self, dataset):
		sample = dataset[0]
		assert len(sample) == 8

	def test_anchor_and_support_shapes_match(self, dataset):
		x, cat, y, qc, igbp, koppen, x_sup, cat_sup = dataset[0]
		assert x.shape == x_sup.shape
		assert cat.shape == cat_sup.shape

	def test_inherits_get_sample(self, dataset):
		# _get_sample should come from the parent, not overridden
		assert "_get_sample" not in dataset.__class__.__dict__
