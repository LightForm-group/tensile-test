"""`tensile_test.tests.test_utils.py`"""

from unittest import TestCase

import numpy as np

from tensile_test.utils import find_nearest_index


class TestUtils(TestCase):

    def test_find_nearest_index(self):
        arr = np.array([1, 2, 3, 4, 5])
        val = 4.3
        self.assertEqual(find_nearest_index(arr, val), 3)
