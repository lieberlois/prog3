"""
Test-Module für body.py
"""

import unittest
from body import Body


class TestBody(unittest.TestCase):
    """
    Class inheriting from unittest.
    This class tests all methods in physics_formula.py
    """

    def setUp(self):
        """
        This Method sets up a body for testing.
        """
        self.body = Body(5.972*10**24)

    def test_body_init(self):
        """
        This Method tests the
        __init__()-function in body.py
        """
        assert self.body.weight == 5.972*10**24

    def test_body_pos(self):
        """
        This Method tests the
        pos-property in body.py
        """
        assert self.body.pos is None

        self.body.pos = (5, 8, 7)
        assert self.body.pos == (5, 8, 7)

    def test_body_speed(self):
        """
        This Method tests the
        speed-property in body.py
        """
        assert self.body.speed is None

        self.body.speed = 30000
        assert self.body.speed == 30000

if __name__ == "__main__":
    unittest.main()
