"""
Test-Module für body.py
"""

from body import Body
import unittest


class Test_Body(unittest.TestCase):
    def test_body_init(self):
        body = Body(5.972*10**24)
        assert body.weight == 5.972*10**24
        
    def test_body_pos(self):
        body = Body(5000)
        assert body.pos == None
        
        body.pos = (5, 8, 7)
        assert body.pos == (5, 8, 7)
        
    def test_body_speed(self):
        body = Body(5000)
        assert body.speed == None
        
        body.speed = 30000
        assert body.speed == 30000
        
if __name__ == "__main__":
    unittest.main()