class Body:
    def __init__(self, weight):
        # TODO:
        self.__pos = None
        self.__speed = None  # if speed is not constant use setter, else like weight
        self.__weight = weight

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, new_pos):
        if new_pos is None:
            return
        else:
            self.__pos = new_pos

    @property
    def speed(self):
        return self.__speed

    @speed.setter
    def speed(self, new_speed):
        if new_speed < 0:
            return
        else:
            self.__speed = new_speed

    @property
    def weight(self):
        return self.__weight

body = Body(5000)
