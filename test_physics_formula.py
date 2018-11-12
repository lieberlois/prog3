"""
Test-Module für physics_formula.py
"""

import unittest
import numpy as np
import physics_formula as phy

class TestPhysicsFormula(unittest.TestCase):
    """
    Class inheriting from unittest.
    This class tests all methods in physics_formula.py
    """
    def test_calc_newton(self):
        """
        This Method tests the
        calc_newton()-function in physics_formula.py
        """
        mass1 = 500
        mass2 = 1000
        mass3 = -500
        acc1 = 20
        acc2 = 50
        acc3 = -10
        assert phy.calc_newton(mass1, acc1) == 10000
        assert phy.calc_newton(mass2, acc2) == 50000

        with self.assertRaises(TypeError):
            phy.calc_newton(mass3, acc3)
            
    def test_calc_gravitational_force(self):
        """
        This Method tests the
        calc_gravitational_force()-function in physics_formula.py
        """
        mass1 = 5.97*10**24
        mass2 = 7.349*10**22
        pos1 = (0, 0, 0)
        pos2 = (384000000, 0, 0)
        directed_force = phy.calc_gravitational_force(mass1, mass2, pos1, pos2)
        assert np.linalg.norm(directed_force) == 1.9851629785156246*10**20


if __name__ == "__main__":
    unittest.main()
