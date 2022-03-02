import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torch


class ElasticDeform(torch.nn.Module):

    def __init__(self, sigma=5, alpha=20):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha

    def forward(self, img):
        # img must be a PIL image
        img = np.array(img)
        img = self.batch_elastic_transform(img, img.shape[0], img.shape[1])
        return img

    def batch_elastic_transform(self, img, height=28, width=28, random_state=None):
        img = img / 255.
        sigma = self.sigma
        alpha = self.alpha
        if random_state is None:
            random_state = np.random.RandomState(None)
        x, y = np.mgrid[0:height, 0:width]

        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        indices = x + dx, y + dy
        img = map_coordinates(img, indices, order=1)
        img = (img * 255)
        return img
