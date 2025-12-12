import numpy as np
import torch
import torch.nn as nn

class lstm(torch.nn.Module):
	def __init__(self, input_dynamic_channels, hidden_dim, output_channels, dropout, layers=1):
		super().__init__()

		self.input_channels = input_dynamic_channels
		self.hidden_dim = hidden_dim
		self.output_channels = output_channels
		self.layers = layers

		self.dynamic_encoder = torch.nn.Linear(in_features=self.input_channels, out_features=self.hidden_dim)
		self.encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.layers, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
		self.dropout = torch.nn.Dropout(p=dropout)


		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic):
		batch, window, _ = x_dynamic.shape

		x_dynamic = self.dynamic_encoder(x_dynamic)
		x_encoder, _ = self.encoder(x_dynamic)
		x_encoder = self.dropout(x_encoder)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)
		return out
    
class ctlstm(torch.nn.Module):
    def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout, layers=1):
        super().__init__()

        self.input_dynamic_channels = input_dynamic_channels
        self.input_static_channels = input_static_channels
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.layers = layers

        self.encoder = torch.nn.LSTM(input_size=self.input_dynamic_channels + self.input_static_channels, hidden_size=self.hidden_dim, num_layers=self.layers, batch_first=True)
        self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, x_static):
        batch, window, _ = x_dynamic.shape

        x = torch.cat((x_dynamic, x_static), dim=-1)
        x_encoder, _ = self.encoder(x)
        x_encoder = self.dropout(x_encoder)
        out = self.out(x_encoder)
        out = out.view(batch, window, self.output_channels)
        return out    
    
class gru(nn.Module):
    def __init__(self, input_dynamic_channels, hidden_dim, output_channels, dropout, layers=1):
        super().__init__()
        
        self.input_dynamic_channels = input_dynamic_channels
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.layers = layers
        
        self.gru = torch.nn.GRU(self.input_dynamic_channels, self.hidden_dim, self.layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_channels)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x_dynamic):
        batch, window, _ = x_dynamic.shape
        
        x_encoder, _ = self.gru(x_dynamic)
        x_encoder = self.dropout(x_encoder)
        out = self.fc(x_encoder)
        out = out.view(batch, window, self.output_channels)
        return out
    
class ctgru(nn.Module):
    def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout, layers=1):
        super().__init__()

        self.input_dynamic_channels = input_dynamic_channels
        self.input_static_channels = input_static_channels
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.layers = layers
        
        self.gru = torch.nn.GRU(self.input_dynamic_channels + self.input_static_channels, self.hidden_dim, self.layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_channels)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x_dynamic, x_static):
        batch, window, _ = x_dynamic.shape

        x = torch.cat((x_dynamic, x_static), dim=-1)
        x_encoder, _ = self.gru(x)
        x_encoder = self.dropout(x_encoder)
        out = self.fc(x_encoder)
        out = out.view(batch, window, self.output_channels)
        return out    