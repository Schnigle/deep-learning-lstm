import torch
import torch.nn.functional as F
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=0):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_embed = (embedding_dim > 0)
        if embedding_dim > 0:
            self.embed_ish = nn.Linear(input_size, embedding_dim)
            self.i2h = nn.Linear(embedding_dim + hidden_size, hidden_size)
        else:
            self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embed = torch.zeros(1, self.input_size).to(hidden.device)
        embed[0, input] = 1
        if self.use_embed:
            embed = self.embed_ish(embed)
        combined = torch.cat((embed, hidden), 1)
        hidden = self.i2h(combined)
        # tanh helps for stability when using higher learning rates
        hidden = self.tanh(hidden)
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, self.hidden_size).to(device)
