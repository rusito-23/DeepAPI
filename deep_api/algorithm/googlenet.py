"""
GoogLeNet Model Extension.
"""

import torchvision
from torch import nn
from torch.hub import load_state_dict_from_url
from utils.exception import ModelInitializationError

WEIGHTS_URL = 'https://download.pytorch.org/models/googlenet-1378be20.pth'


class DeepGoogLeNet(torchvision.models.GoogLeNet):
    """
    GoogLeNet Extension to save hooks for the Deep Dream Algorithm.
    It will pass an input through a given number of layers and save
    the partial outputs into the `features` array.
    Parameters:
        - loi:
            Layers Of Interest - pair(a, b) with the layer indexes to lookup
    """
    features = []
    hooks = []

    def __init__(self, loi):
        super(DeepGoogLeNet, self).__init__()
        # load pretrained weights
        state_dict = load_state_dict_from_url(WEIGHTS_URL, progress=False)
        self.load_state_dict(state_dict)

        # loi validation
        la, lb = loi
        if lb <= 0 or lb > 16:
            raise ModelInitializationError()

        # prepare layers
        named_children = dict(self.named_children())
        layer_names = list(named_children.keys())[:lb]
        layers = [named_children[name] for name in layer_names]

        # set layers and hooks
        self.features = []
        self.layers = nn.Sequential(*layers)
        self.hooks = [layer.register_forward_hook(self.feature_hook)
                      for layer in self.layers[la:lb]]

    def feature_hook(self, module, _in, _out):
        self.features.append(_out)

    def remove(self):
        [hook.remove() for hook in self.hooks]

    def forward(self, x):
        self.features = []
        return self.layers(x)
