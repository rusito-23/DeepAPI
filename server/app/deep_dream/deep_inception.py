"""
Deep Dream Base Model extending the InceptionV3 Pretrained Model.
"""

import torchvision


class DeepInception3(torchvision.models.Inception3):
    """
    Inception Net v3 Extension for Deep Dream.
    Params:
        - loi: layers of interest: names of the layers that will be saved
                with hooks to get their outputs.
        - weights_path: pretrained weights path
        - **kwargs: same args as in torchvision.models.Inception3
    """
    features = []
    hooks = []

    def __init__(self, loi, weights_path, **kwargs):
        super(DeepInception3, self).__init__(**kwargs)

        # load pretrained weights
        self.load_state_dict(weights_path)

        # prepare hooks to save features
        for layer_name in loi:
            layer = dict(self.named_children())[layer_name]
            layer_hook = layer.register_forward_hook(self.feature_hook)
            self.hooks.append(layer_hook)

    def feature_hook(self, module, _in, _out):
        self.features.append(_out)

    def remove(self):
        [hook.remove() for hook in self.hooks]

    def forward(self, x):
        self.features = []
        return self.layers(x)
