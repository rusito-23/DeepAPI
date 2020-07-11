"""
Unit Tests Runner
"""

import os
import sys
import unittest
sys.path.append(os.path.join(os.getcwd(), 'deep_api'))

if __name__ == '__main__':
    suite = unittest.TestLoader().discover('./test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
