"""
Hilfs-Module für physikalische Formeln
"""
import numpy as np

G = 6.672*10**(-11)


def calc_newton(mass, acceleration):
    """
    Funktion zur Berechnung der Kraft F nach Newton

    :param mass: Masse des Körpers
    :param acceleration: Beschleunigung des Körpers
    :return: Newton Kraft
    """
    return mass*acceleration


def calc_gravitational_force(mass1, mass2, position1, position2):
    """
    Brechnet die Gravitionskraft, mit der eine Punktmasse mass1, die sich an der Position position1 befindet,
    von einer Punktmasse mass2, die sich an der Position position2 befindet, angezogen wird

    :param mass1: Punktmasse eines Körpers 1
    :param mass2: Punktmasse eines zweiten Körpers 2
    :param position1: Position von Körper 1
    :param position2: Position von Körper 2
    :return: Gravitationskraft als Vektor
    """
    pos1 = np.array(position1)
    pos2 = np.array(position2)
    vector_length = np.linalg.norm(pos2-pos1)
    return G*((mass1*mass2)/vector_length**3)*(pos2-pos1)
