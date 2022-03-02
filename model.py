import torch
import torch.nn as nn
import math


class PrototypeLayer(nn.Module):

    def __init__(self, num_features, num_prototypes):
        super().__init__()
        self.num_features = num_features
        self.num_prototypes = num_prototypes

        self.prototype_weights = nn.Parameter(torch.empty((num_prototypes, num_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.prototype_weights)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.unsqueeze(1)
        return torch.sqrt(torch.sum((inp - self.prototype_weights) ** 2, -1))


class Autoencoder(nn.Module):

    def __init__(self, output_map_sizes_list, filter_sizes_list, stride_sizes_list, activation, input_channels=1):
        super().__init__()
        self.activation = activation
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        last_channel_size = input_channels
        for output_map_size, filter_size, stride_size in zip(output_map_sizes_list, filter_sizes_list,
                                                             stride_sizes_list):
            self.encoder.append(
                nn.Conv2d(last_channel_size, output_map_size, filter_size, stride=stride_size, padding=(1, 1)))
            self.decoder.insert(0, nn.ConvTranspose2d(output_map_size, last_channel_size, filter_size,
                                                      stride=stride_size, padding=(1, 1)))
            last_channel_size = output_map_size

    def forward(self, x):
        shapes = []
        for layer in self.encoder:
            shapes.append(tuple(x.shape))
            x = layer(x)
            x = self.activation(x)
        latent_state = x
        for layer, shape in zip(self.decoder, shapes[::-1]):
            x = layer(x, output_size=shape)
            x = self.activation(x)
        reconstruction = x

        return latent_state, reconstruction

    def reconstruct(self, latent_state):
        x = latent_state
        # Hard coding output shapes for now :/
        for layer, shape in zip(self.decoder, [(1, 32, 4, 4), (1, 32, 7, 7), (1, 32, 14, 14), (1, 1, 28, 28)]):
            x = layer(x, output_size=shape)
            x = self.activation(x)
        reconstruction = x
        return reconstruction


class PrototypeNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.autoencoder = Autoencoder(config['output_map_sizes'], config['filter_sizes'], config['stride_sizes'],
                                       activation=nn.Sigmoid())
        self.prototype_layer = PrototypeLayer(config['feature_size'], config['num_prototypes'])
        self.fc = nn.Linear(config['num_prototypes'], config['num_classes'])

    def forward(self, input_image_batch: torch.Tensor):
        latent_state, reconstruction = self.autoencoder(input_image_batch)
        x = torch.flatten(latent_state, 1)
        x = self.prototype_layer(x)
        prototype_out = x
        x = self.fc(x)

        return x, latent_state, reconstruction, prototype_out

    @staticmethod
    def get_default_config():
        config = {
            'num_prototypes': 15,
            'output_map_sizes': [32, 32, 32, 10],
            'stride_sizes': [2, 2, 2, 2],
            'filter_sizes': [3, 3, 3, 3],
            'feature_size': 40,
            'batch_size': 250,
            'num_classes': 10,
            'lambda': 0.05,
            'lambda1': 0.05,
            'lambda2': 0.05,
            'lr': 0.0001
        }

        return config
