"""
Test-Module für physics_formula.py
"""

import physics_formula as phy
import unittest
import numpy as np


class TestPhysicsFormula(unittest.TestCase):
    def test_calc_newton(self):
        mass1 = 500
        mass2 = 1000
        mass3 = -500
        acc1 = 20
        acc2 = 50
        acc3 = -10
        assert 10000 == phy.calc_newton(mass1, acc1)
        assert 50000 == phy.calc_newton(mass2, acc2)
        
        with self.assertRaises(Exception):
            phy.calc_newton(mass3, acc3)
            
    def test_calc_gravitational_force(self):
        mass1 = 5.97*10**24  
        mass2 = 7.349*10**22
        pos1 = (0, 0, 0)
        pos2 = (384000000, 0, 0)
        directed_force = phy.calc_gravitational_force(mass1, mass2, pos1, pos2)
        assert np.linalg.norm(directed_force) == 1.9851629785156246*10**20


if "__main__" == __name__:
    unittest.main()
