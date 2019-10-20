import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_dim):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError('{} is not an appropriate attention method.'.format(self.method))
        self.hidden_dim = hidden_dim
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_dim, hidden_dim)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
            self.v = nn.Parameter(torch.tensor(hidden_dim).float())
        self.W_cat = nn.Linear(self.hidden_dim*2, hidden_dim)

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(
            encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        if isinstance(hidden, tuple):  # if hidden from lstm
            hidden = hidden[1]  # only consider the cell state
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_dim dimensions
        attn_energies = attn_energies.masked_fill(mask == 0, -1e10).t()

        # Return the softmax normalized probability scores with added dimension
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
