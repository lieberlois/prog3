"""
Modul zur physikalischen Berechnung von Kräften
"""
import numpy as np
from simulation_constants import G_CONSTANT


def calc_newton(mass, acceleration):
    """
    Funktion zur Berechnung der Kraft F nach Newton

    :param mass: Masse des Körpers
    :param acceleration: Beschleunigung des Körpers
    :return: Newton Kraft
    """
    if mass < 0:
        raise Exception
    return mass*acceleration


def calc_gravitational_force(mass1, mass2, position1, position2):
    """
    Brechnet die Gravitionskraft, mit der eine Punktmasse mass1, die sich an der Position position1 be
findet,
    von einer Punktmasse mass2, die sich an der Position position2 be
findet, angezogen wird

    :param mass1: Punktmasse eines Körpers 1
    :param mass2: Punktmasse eines zweiten Körpers 2
    :param position1: Position von Körper 1
    :param position2: Position von Körper 2
    :return: Gravitationskraft als Vektor
    """
    
    # TODO:A body might be able to contain its mass position and current speed,
    #     so its probably better to access bodies of using these parameters.
    pos1 = np.array(position1)
    pos2 = np.array(position2)
    vector_length = np.linalg.norm(pos2-pos1)
    return G_CONSTANT*((mass1*mass2)/vector_length**3)*(pos2-pos1)
