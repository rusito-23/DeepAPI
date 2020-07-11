"""
GoogLeNet Model Deep Dream Extension Tests.
"""

import torch
from algorithm.googlenet import DeepGoogLeNet
from test_case import TestCase
from utils.exception import (PreProcessingError,
                             PostProcessingError,
                             ModelInitializationError)


class TestGoogLeNet(TestCase):

    def test_init(self):
        self.assertNotRaise(lambda: DeepGoogLeNet((0, 4)),
                            'Model Initalization failed')

    def test_init_neg_a(self):
        self.assertRaises(ModelInitializationError,
                          lambda: DeepGoogLeNet((-1, 9)))

    def test_init_neg_b(self):
        self.assertRaises(ModelInitializationError,
                          lambda: DeepGoogLeNet((1, -9)))

    def test_init_over_b(self):
        self.assertRaises(ModelInitializationError,
                          lambda: DeepGoogLeNet((1, 17)))

    def test_init_loi_interpretation(self):
        model = DeepGoogLeNet((2, 4))
        assert len(model.hooks) == 2, 'Incorrect hook count'
        assert len(model.layers) == 4, 'Incorrect layer count'

    def test_feature_extraction_count(self):
        model = DeepGoogLeNet((2, 4))
        inp = torch.randn(1, 3, 256, 256)
        _ = model(inp)

        assert len(model.features) == 2, 'Incorrect feature count'
