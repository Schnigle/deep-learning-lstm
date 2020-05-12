import torch
import torch.nn.functional as F
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, embedding_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.i2h = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embed = self.embedding(input)
        combined = torch.cat((embed, hidden), 1)
        hidden = self.i2h(combined)
        # tanh helps for stability when using higher learning rates
        hidden = self.tanh(hidden)
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, self.hidden_size).to(device)
