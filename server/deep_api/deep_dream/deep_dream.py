"""
Deep Dream Algorithm.
"""

import torch
from PIL import Image, ImageFilter
from deep_dream.deep_inception import DeepInception3
import deep_dream.deep_utils as utils


class DeepDream:
    """
    Deep Dream Algorithm.
    """

    def __init__(self, loi, weights_path):
        self.model = DeepInception3(loi=loi, weights_path=weights_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

    def deep_dream_loss(self, model, target):
        """
        Deep Dream Loss:
        Uses the model features (declared when setting the model hooks)
        and uses the sum of the mean values of these features as loss.
        """
        # run the image through the net
        _ = deep_dream(target)

        # get the loss
        losses = [torch.mean(feat) for feat in model.features]
        loss = torch.stack(losses, axis=0).sum()
        return loss

    def dream_inception(self,
                        image,
                        epochs,
                        learning_rate,
                        learning_weight):
        """
        Deep Dream Inception.
        Passes the given image through the model and uses the
        Gradient Ascent Method to update the image.
        """
        target = utils.preprocess(image).to(self.device)
        for _ in range(epochs):
            # reset gradient
            if target.grad is not None:
                target.grad.zero_()

            # loss backward
            loss = self.deep_dream_loss(deep_dream, target)
            loss.backward(retain_graph=True)

            # gradient ascent step (standarizing the gradient)
            grad = target.grad.data / (torch.std(target.grad.data) + 1e-8)
            learning_rate = learning_rate / learning_weigh
            target.data = target.data + grad * learning_rate

            # clip pixel values
            target.data = utils.clip(target.data)
        return target

    def create_inceptions(self,
                          image,
                          n_inceptions,
                          scale_factor,
                          blur_radius):
        """
        Creates the deep dream inceptions given the original image.
        In order to get the best result for the Deep Dream Algorithm,
        we need to create different samples, using the original image
        downscaled and blured.
        """
        inceptions = [image]
        for i in range(n_inceptions - 1):
            inception = inceptions[-1]
            inception = inception.filter(ImageFilter.GaussianBlur(blur_radius))
            inception = scaled(inception, scale_factor)
            inceptions.append(blurred_scaled)
        return inceptions

    def __call__(self, source_image,
                 epochs=100,
                 learning_rate=1,
                 loi=['Mixed_5b', 'Mixed_6b'],
                 n_inceptions=12,
                 scale_factor=0.7,
                 blend_factor=0.3,
                 blur_radius=60):
        """
        This is the main call to perform the Deep Dream algorithm,
        it creates the inceptions (original image downscaled and blured)
        and optimizes each of these running them through the model.
        Each inception is also blended with the following inception.
        """

        # create inceptions
        inceptions = self.create_inceptions(image,
                                            n_inceptions,
                                            scale_factor,
                                            blur_radius)

        # run a step for each of these
        for w, inception in enumerate(inceptions[::-1]):
            if 'dream' not in vars() or 'dream' not in globals():
                # upsample and blend last inception
                dream = dream.resize(inception.size)
                scale = Image.blend(inception, dream, blend_factor)

            # run the step
            learning_weight = w + 1
            dream = self.dream_inception(model,
                                         inception,
                                         learning_weight)

            # re convert to PIL and save for the next it
            dream = dream.cpu().clone().detach().squeeze(0)
            dream = utils.postprocess(dream)

        return dream
