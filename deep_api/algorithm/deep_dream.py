"""
Deep Dream Algorithm.
"""

from time import time
import logging
import torch
from torch.hub import load_state_dict_from_url
from algorithm.googlenet import DeepGoogLeNet, WEIGHTS_URL
import algorithm.im_utils as utils
from utils.exception import (
        UnknownStyle,
        PreProcessingError,
        PostProcessingError
)

logger = logging.getLogger('DEEP_API')


class DeepDream:
    """
    Deep Dream Algorithm.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda'
                                   if torch.cuda.is_available()
                                   else 'cpu')
        _ = load_state_dict_from_url(WEIGHTS_URL, progress=False)
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
        style = self.cfg.STYLES_CFG[style_name]
        logger.info(f'Running deep dream algorithm using style: {style_name}')

        # prepare model
        model = DeepGoogLeNet(loi=style.loi)
        model.eval()

        # prepare input image
        try:
            original_size = image.size
            image = image.resize((style.size, style.size))
            target = utils.preprocess(image).to(self.device)
        except Exception:
            raise PreProcessingError()

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
            target.data = utils.clip(target.data)

            logger.debug(f'Epoch {e}/{style.epochs} '
                         f'took: {time() - start:.2f}')

        logger.info('Deep Dream with style: '
                    f'{style_name} took: {time() - start:.2f}')

        try:
            dream = target.cpu().clone().detach().squeeze(0)
            dream = utils.postprocess(dream).resize(original_size)
        except Exception:
            raise PostProcessingError()

        return dream
