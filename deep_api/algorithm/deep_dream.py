"""
Deep Dream Algorithm.
"""

from time import time
import torch
from torch import nn
import torchvision
import logging
from torchvision import transforms as T
from PIL import Image, ImageFilter
from utils.exception import UnknownStyle
from torch.hub import load_state_dict_from_url

logger = logging.getLogger('DEEP_API')

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMEAN = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
ISTD = [1/0.229, 1/0.224, 1/0.225]
WEIGHTS_URL = 'https://download.pytorch.org/models/googlenet-1378be20.pth'


def preprocess(image):
    """
    Preprocesses the image to be fed to the Deep Model Algorithm.
    Resizes, normalizes the image and performs Pytorch Tensor Conversion.
    """
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transforms(image).unsqueeze(0).requires_grad_(True)


def postprocess(tensor):
    """
    Processes the output for the Deep Dream Algorithm.
    """
    transforms = T.Compose([
        T.Normalize(IMEAN, ISTD),
        T.ToPILImage()
    ])
    return transforms(tensor)


def clip(image):
    """
    Set the image pixel values between 0 and 1.
    """
    for c, (mean, std) in enumerate(zip(MEAN, STD)):
        image[0, c] = torch.clamp(image[0, c], -mean/std, (1 - mean)/std)
    return image


class DeepGoogLeNet(torchvision.models.GoogLeNet):
    """
    GoogLeNet Extension to save hooks for the Deep Dream Algorithm.
    It will pass an input through a given number of layers and save
    the partial outputs into the `features` array.
    Parameters:
        - loi:
            Layers Of Interest - the first n layers to save hooks
    """
    features = []
    hooks = []

    def __init__(self, loi):
        super(DeepGoogLeNet, self).__init__()

        # load pretrained weights
        state_dict = load_state_dict_from_url(WEIGHTS_URL, progress=False)
        self.load_state_dict(state_dict)

        # check layers of interest
        assert loi > 0, 'Inavlid loi, must be > 0'
        assert loi <= 16, 'Invalid loi, only first 16 are available.'

        # prepare layers
        named_children = dict(self.named_children())
        layer_names = list(named_children.keys())[:loi]
        layers = [named_children[name] for name in layer_names]
        self.layers = nn.Sequential(*layers)

        # prepare hooks
        for layer in self.layers:
            self.hooks.append(layer.register_forward_hook(self.feature_hook))

    def feature_hook(self, module, _in, _out):
        self.features.append(_out)

    def remove(self):
        [hook.remove() for hook in self.hooks]

    def forward(self, x):
        self.features = []
        return self.layers(x)


class DeepDream:
    """
    Deep Dream Algorithm.
    """

    def __init__(self, cfg):
        self.styles = cfg.styles
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        logger.info('Deep Dream Class Initialized succesfully')

    def deep_dream_loss(self, model, target):
        """
        Deep Dream Loss:
        Uses the model features (declared when setting the model hooks)
        and uses the sum of the mean values of these features as loss.
        """
        # run the image through the net
        _ = model(target)

        # get the loss
        losses = [torch.mean(feat) for feat in model.features]
        loss = torch.stack(losses, axis=0).sum()
        return loss

    def __call__(self, image, style_name):
        """
        Deep Dream Main Algorithm.
        Passes the given image through the model and uses the
        Gradient Ascent Method to update the image.
        """
        # prepare style
        if style_name not in self.styles:
            raise UnknownStyle(style_name)
        style = self.styles[style_name]
        logger.info(f'Running deep dream algorithm using style: {style_name}')

        # prepare model
        model = DeepGoogLeNet(loi=style.loi)
        model.eval()

        # prepare input image
        original_size = image.size
        image = image.resize((style.size, style.size))
        target = preprocess(image).to(self.device)

        start = time()
        for e in range(style.epochs):
            # reset gradient
            if target.grad is not None:
                target.grad.zero_()

            # loss backward
            loss = self.deep_dream_loss(model, target)
            loss.backward(retain_graph=True)

            # gradient ascent step (standarizing the gradient)
            grad = target.grad.data / (torch.std(target.grad.data) + 1e-8)
            target.data = target.data + grad * style.learning_rate

            # clip pixel values
            target.data = clip(target.data)

            logger.debug(f'Epoch {e}/{style.epochs} '
                         f'took: {time() - start:.2f}')

        logger.info('Deep Dream with style: '
                    f'{style_name} took: {time() - start:.2f}')
        dream = target.cpu().clone().detach().squeeze(0)
        dream = postprocess(dream).resize(original_size)
        return dream
