import numpy as np
import torch
import torch.nn as nn

class lstm(torch.nn.Module):
	def __init__(self, input_dynamic_channels, hidden_dim, output_channels, dropout, layers=1, **kwargs):
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

	def forward(self, x_dynamic, **kwargs):
		batch, window, _ = x_dynamic.shape

		x_dynamic = self.dynamic_encoder(x_dynamic)
		x_encoder, _ = self.encoder(x_dynamic)
		x_encoder = self.dropout(x_encoder)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)
		return out

class ctlstm_decoder(torch.nn.Module):
    def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout, layers=1):
        super().__init__()

        self.input_dynamic_channels = input_dynamic_channels
        self.input_static_channels = input_static_channels
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.layers = layers
        
        self.dynamic_encoder = torch.nn.Linear(in_features=self.input_dynamic_channels, out_features=self.hidden_dim // 2)
        self.static_encoder = torch.nn.Linear(in_features=self.input_static_channels, out_features=self.hidden_dim // 2)
        self.encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.layers,batch_first=True)
        self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.relu = torch.nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, x_static):
        batch, window, _ = x_dynamic.shape
        
        x_dynamic_encoder = self.relu(self.dynamic_encoder(x_dynamic))
        x_static_encoder = self.relu(self.static_encoder(x_static))
        
        x = torch.cat((x_dynamic_encoder, x_static_encoder), dim=-1)
        x_encoder, _ = self.encoder(x)
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
        
        self.dynamic_encoder = torch.nn.Linear(in_features=self.input_dynamic_channels, out_features=self.hidden_dim // 2)
        self.static_encoder = torch.nn.Linear(in_features=self.input_static_channels, out_features=self.hidden_dim // 2)
        self.encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.layers, batch_first=True)
        self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, x_static):
        batch, window, _ = x_dynamic.shape
        
        x_dynamic = self.dynamic_encoder(x_dynamic)
        x_static = self.static_encoder(x_static)
        
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
        
        self.dynamic_encoder = torch.nn.Linear(in_features=self.input_dynamic_channels, out_features=self.hidden_dim // 2)
        self.static_encoder = torch.nn.Linear(in_features=self.input_static_channels, out_features=self.hidden_dim // 2)
        self.gru = torch.nn.GRU(self.hidden_dim, self.hidden_dim, self.layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_channels)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x_dynamic, x_static):
        batch, window, _ = x_dynamic.shape
        
        x_dynamic = self.dynamic_encoder(x_dynamic)
        x_static = self.static_encoder(x_static)
        
        x = torch.cat((x_dynamic, x_static), dim=-1)
        x_encoder, _ = self.encoder(x)
        x_encoder = self.dropout(x_encoder)
        out = self.out(x_encoder)
        out = out.view(batch, window, self.output_channels)
        return out    

class tamlstm(torch.nn.Module):
    def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout, layers=1):
        super().__init__()

        # PARAMETERS
        self.input_dynamic_channels = input_dynamic_channels
        self.input_static_channels = input_static_channels
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.layers = layers
        # LAYERS
        self.dynamic_encoder = torch.nn.Linear(in_features=self.input_dynamic_channels, out_features=self.hidden_dim)
        self.static_encoder = torch.nn.Linear(in_features=self.input_static_channels, out_features=self.hidden_dim)
        self.film_layer_head = torch.nn.Linear(input_static_channels, 2 * self.hidden_dim)
        self.encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.layers,batch_first=True)
        self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.relu = torch.nn.ReLU()

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, x_static):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x_dynamic_encoder = self.dynamic_encoder(x_dynamic)
        x_static_encoder = self.relu(self.static_encoder(x_static))
        x_static_encoder = x_static_encoder + torch.ones_like(x_static_encoder)

        x = x_static_encoder*x_dynamic_encoder+x_static_encoder
        x_encoder, _ = self.encoder(x)
        embedding = self.film_layer_head(x_static)        
        gammas, betas = torch.split(embedding, x_encoder.shape[-1], dim=-1)      
        gammas = gammas + torch.ones_like(gammas)
        x_encoder = x_encoder * gammas + betas  
        x_encoder = self.dropout(x_encoder)
        out = self.out(x_encoder)
        out = out.view(batch, window, self.output_channels)

        return out   
    
class ae_tamrl(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, code_dim, output_channels):
        super().__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.output_channels = output_channels
        self.device = torch.device("cuda:0")

        # LAYERS
        self.instance_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_channels, out_features=self.hidden_dim),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.LeakyReLU(0.2)
        )
        self.temporal_encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bidirectional=True, batch_first=True)	# AE
        self.code_linear = torch.nn.Linear(self.hidden_dim, self.code_dim)																		# AE
        self.decode_linear = torch.nn.Linear(self.code_dim, self.hidden_dim)																	# AE
        self.temporal_decoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)						# AE
        self.instance_decoder = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.input_channels)									# AE
        self.static_out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):

        # GET SHAPES
        batch, window, _ = x.shape

        # OPERATIONS

        x_encoder = self.instance_encoder(x.view(-1, self.input_channels)).view(batch, window, -1)		# ENCODE
        _, x_encoder = self.temporal_encoder(x_encoder)													# ENCODE
        enc_vec = torch.sum(x_encoder[0], dim=0)														# ENCODE

        z = self.code_linear(enc_vec)																	# CODE_VEC

        static_out = self.static_out(z)																	# STATIC DECODE

        decode_vec = self.decode_linear(z)																# DECODE
        out = torch.zeros(batch, window, self.input_channels).to(self.device)							# DECODE
        input = torch.unsqueeze(torch.zeros_like(decode_vec), dim=1)									# DECODE
        h = (torch.unsqueeze(decode_vec, dim=0), torch.unsqueeze(torch.zeros_like(decode_vec), dim=0))	# DECODE
        for step in range(window):																		# DECODE
            input, h = self.temporal_decoder(input, h)													# DECODE
            out[:,step] = self.instance_decoder(input.squeeze())										# DECODE

        return z, enc_vec, static_out, out