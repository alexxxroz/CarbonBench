import torch
import pytest

import carbonbench


BATCH = 4
SEQ_LEN = 30
DYN_CH = 10
STATIC_CH = 51
OUT_CH = 3
HIDDEN = 32
STRIDE = 15


class TestLSTM:
	def test_forward_shape(self):
		model = carbonbench.lstm(
			input_dynamic_channels=DYN_CH, hidden_dim=HIDDEN,
			output_channels=OUT_CH, dropout=0.1, layers=1
		)
		x = torch.randn(BATCH, SEQ_LEN, DYN_CH)
		out = model(x)
		assert out.shape == (BATCH, SEQ_LEN, OUT_CH)


class TestCTLSTM:
	def test_forward_shape(self):
		model = carbonbench.ctlstm(
			input_dynamic_channels=DYN_CH, input_static_channels=STATIC_CH,
			hidden_dim=HIDDEN, output_channels=OUT_CH, dropout=0.1, layers=1
		)
		x = torch.randn(BATCH, SEQ_LEN, DYN_CH)
		x_static = torch.randn(BATCH, SEQ_LEN, STATIC_CH)
		out = model(x, x_static)
		assert out.shape == (BATCH, SEQ_LEN, OUT_CH)


class TestGRU:
	def test_forward_shape(self):
		model = carbonbench.gru(
			input_dynamic_channels=DYN_CH, hidden_dim=HIDDEN,
			output_channels=OUT_CH, dropout=0.1, layers=1
		)
		x = torch.randn(BATCH, SEQ_LEN, DYN_CH)
		out = model(x)
		assert out.shape == (BATCH, SEQ_LEN, OUT_CH)


class TestCTGRU:
	def test_forward_shape(self):
		model = carbonbench.ctgru(
			input_dynamic_channels=DYN_CH, input_static_channels=STATIC_CH,
			hidden_dim=HIDDEN, output_channels=OUT_CH, dropout=0.1, layers=1
		)
		x = torch.randn(BATCH, SEQ_LEN, DYN_CH)
		x_static = torch.randn(BATCH, SEQ_LEN, STATIC_CH)
		out = model(x, x_static)
		assert out.shape == (BATCH, SEQ_LEN, OUT_CH)


class TestTransformer:
	def test_forward_shape(self):
		model = carbonbench.transformer(
			input_dynamic_channels=DYN_CH, input_static_channels=STATIC_CH,
			output_channels=OUT_CH, seq_len=SEQ_LEN, hidden_dim=HIDDEN,
			nhead=4, num_layers=2, dropout=0.1
		)
		x = torch.randn(BATCH, SEQ_LEN, DYN_CH)
		x_static = torch.randn(BATCH, SEQ_LEN, STATIC_CH)
		out = model(x, x_static)
		assert out.shape == (BATCH, SEQ_LEN, OUT_CH)


class TestPatchTransformer:
	def test_forward_shape(self):
		model = carbonbench.patch_transformer(
			input_dynamic_channels=DYN_CH, input_static_channels=STATIC_CH,
			output_channels=OUT_CH, seq_len=SEQ_LEN, pred_len=STRIDE,
			patch_len=4, stride=4, hidden_dim=HIDDEN,
			nhead=4, num_layers=2, dropout=0.1
		)
		x = torch.randn(BATCH, SEQ_LEN, DYN_CH)
		x_static = torch.randn(BATCH, SEQ_LEN, STATIC_CH)
		out = model(x, x_static)
		assert out.shape == (BATCH, STRIDE, OUT_CH)


class TestTAMLSTM:
	def test_forward_shape(self):
		model = carbonbench.tamlstm(
			input_dynamic_channels=DYN_CH, input_static_channels=STATIC_CH,
			hidden_dim=HIDDEN, output_channels=OUT_CH, dropout=0.1, layers=1
		)
		x = torch.randn(BATCH, SEQ_LEN, DYN_CH)
		x_static = torch.randn(BATCH, SEQ_LEN, STATIC_CH)
		out = model(x_dynamic=x, x_static=x_static)
		assert out.shape == (BATCH, SEQ_LEN, OUT_CH)


class TestAETAMRL:
	def test_forward_shape(self):
		in_ch = DYN_CH + STATIC_CH
		model = carbonbench.ae_tamrl(
			input_channels=in_ch, hidden_dim=32,
			code_dim=16, output_channels=16
		)
		x = torch.randn(BATCH, SEQ_LEN, in_ch)
		z, enc_vec, static_out, recon = model(x)
		assert z.shape == (BATCH, 16)
		assert enc_vec.shape == (BATCH, 32)
		assert static_out.shape == (BATCH, 16)
		assert recon.shape == (BATCH, SEQ_LEN, in_ch)
