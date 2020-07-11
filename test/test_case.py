"""
Base Test Case.
"""

import os
import tempfile
import shutil
from unittest import TestCase as BaseTestCase
from cfg.main_cfg import CFG


class TestCase(BaseTestCase):
    cfg = CFG()
    EXAMPLES_PATH = './test/examples'

    """ Test Case Methods """

    @classmethod
    def setUpClass(cls):
        cls.cfg.OUTPUT_FOLDER = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cfg.OUTPUT_FOLDER)

    """ Utils """

    def assertNotRaise(self, fun, message):
        try:
            fun()
        except Exception:
            raise AssertionError(message)
