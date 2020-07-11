"""
Test Deep Dream Algorithm Full Functionality.
"""

from PIL import Image
import numpy as np
from mockito import when2, unstub, ANY
from algorithm.deep_dream import DeepDream
from test_case import TestCase
from algorithm import im_utils
from utils.exception import (
    PreProcessingError,
    PostProcessingError,
    UnknownStyle,
)


class TestDeepDreamAlgo(TestCase):

    def test_init_success(self):
        self.assertNotRaise(lambda: DeepDream(self.cfg),
                            'Deep Dream Initialization Error')

    def test_invalid_style_name(self):
        algo = DeepDream(self.cfg)
        self.assertRaises(UnknownStyle, lambda: algo(None, 'wrong_name'))

    def test_input_none(self):
        algo = DeepDream(self.cfg)
        self.assertRaises(PreProcessingError, lambda: algo(None, 'trippy'))

    def test_input_invalid(self):
        algo = DeepDream(self.cfg)
        inp = np.random.randn(3, 224, 224)
        self.assertRaises(PreProcessingError, lambda: algo(inp, 'trippy'))

    def test_output_invalid(self):
        algo = DeepDream(self.cfg)
        inp = Image.new('RGB', (256, 256))
        when2(im_utils.postprocess, ANY).thenRaise(Exception('EXC'))
        self.assertRaises(PostProcessingError, lambda: algo(inp, 'trippy'))
