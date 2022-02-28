import torch
import torch.nn as nn

import numpy as np

class PrototypeLayer(nn.Module):

    def __init__(self, num_features, num_prototypes):
        self.num_features = num_features
        self.num_prototypes = num_prototypes

        self.prototype_weights = nn.Parameter(torch.empty((num_prototypes, num_features)))

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.prototype_weights, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum((input - self.prototype_weights) ** 2))


class Autoencoder(nn.Module):

    def __init__(self, output_map_sizes_list, filter_sizes_list, stride_sizes_list, activation):

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for output_map_size, filter_size, stride_size in zip(output_map_sizes_list, filter_sizes_list, stride_sizes_list):
            self.encoder.append(nn.Conv2d(output_map_size, filter_size, strides=stride_sizes_list))
            self.decoder.insert(0, nn.Conv2dTranpose(output_map_size, filter_size, strides=stride_sizes_list))

    def forward(self, input):
        pass

    def forward_encoder(self, input):
        pass

    def forward_decoder(self, latent_state):
        pass

class PrototypeNetwork(nn.Module):

    def __init__(self, config):

        self.autoencoder = Autoencoder(config['output_map_sizes'], config['filter_sizes'], config['stride_sizes'], activation=nn.Sigmoid())
        self.prototype_layer = PrototypeLayer(config['feature_size'], config['num_prototypes'])
        #self.fc = nn.Linear():

    def forward(self, input_image_batch: torch.Tensor):
        pass
