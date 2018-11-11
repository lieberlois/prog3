"""
This module brings the Body-Class.
Each body will be represented in the simulation.
"""

class Body:
    """
    This class contains all important information
    about a body.
    """
    def __init__(self, weight):
        self.__pos = None
        self.__speed = None
        self.__weight = weight

    @property
    def pos(self):
        """
        This method gets the value
        of the pos-property
        """
        return self.__pos

    @pos.setter
    def pos(self, new_pos):
        """
        This method sets the value
        of the pos-property
        """
        if new_pos is None:
            return
        self.__pos = new_pos

    @property
    def speed(self):
        """
        This method gets the value
        of the speed-property
        """
        return self.__speed

    @speed.setter
    def speed(self, new_speed):
        """
        This method sets the value
        of the speed-property
        """
        if new_speed < 0:
            return
        self.__speed = new_speed

    @property
    def weight(self):
        """
        This method gets the value
        of the weight-property
        """
        return self.__weight
