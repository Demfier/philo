import torch
from torch import nn
from torch import optim
from torch.nn import Module, Linear, BCEWithLogitsLoss, Sequential, BatchNorm1d, LeakyReLU


class GanDiscriminator(Module):
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim: dimension of the input latent codes
        hidden_dim: classifier's hidden dimension
        """
        super(GanDiscriminator, self).__init__()
        self._classifier = Sequential(
            Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), LeakyReLU(0.2),
            Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), LeakyReLU(0.2),
            Linear(hidden_dim, 1)  # simple binary classification, hence 1
        )
        self._loss = BCEWithLogitsLoss()
        self._optimizer = optim.Adam(self.parameters(), config['lr'])


    def optimize(self, clip):
        nn.utils.clip_grad_norm_(self.parameters(), clip)
        self._optimizer.step()


    def forward(self, input_latent, labels=None):
        output = self._classifier(input_latent)
        output_dict = {'output': output}
        if labels is not None:
            output_dict['loss'] = self._loss(output, labels)
            output_dict['accuracy'] = (torch.sigmoid(ouput).round() == labels).float().mean()
        return output_dict


class GanGenerator(Module):
    def __init__(self, latent_dim, temperature=0.01):
        super(GanGenerator, self).__init__()
        self._latent_mapper = Sequential(
            Linear(latent_dim, 2*latent_dim), BatchNorm1d(2*latent_dim), LeakyReLU(0.2),
            Linear(2*latent_dim, 2*latent_dim), BatchNorm1d(2*latent_dim), LeakyReLU(0.2),
            Linear(2*latent_dim, latent_dim)
        )

        self._temperature = temperature
        self._ce_loss = BCEWithLogitsLoss()
        self._optimizer = optim.Adam(self.parameters(), config['lr'])

    def add_noise(self, latent):
        """
        Adds standard normal noise to a latent vector
        """
        return self._temperature*torch.randn(latent.shape) + latent

    def optimize(self, clip):
        nn.utils.clip_grad_norm_(self.parameters(), clip)
        self._optimizer.step()

    def forward(self, target_latent, discriminator):
        # add noise to input data
        target_latent = self.add_noise(target_latent)
        z_g = self._latent_mapper(target_latent)
        output_dict = {}
        # continue discriminator part
        if discriminator is not None:
            predicted = discriminator(z_g)['output']
            # the generator wants the disc to think these as true
            desired = torch.ones_like(predicted)
            output_dict['loss'] = self._ce_loss(predicted, desired)
            output_dict['accuracy'] = (torch.sigmoid(predicted).round() == desired).float().mean()

        output_dict.update({
            'z_g': z_g
            })
        return output_dict
