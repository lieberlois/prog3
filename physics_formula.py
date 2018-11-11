"""
Modul zur physikalischen Berechnung von Kraeften
"""
import numpy as np
from simulation_constants import G_CONSTANT


def calc_newton(mass, acceleration):
    """
    Funktion zur Berechnung der Kraft F nach Newton

    params:
        mass: Masse des Koerpers
        acceleration: Beschleunigung des Koerpers
    return:
        Newton Kraft
    """
    if mass <= 0:
        raise TypeError('Mass has to be greater than 0')
    return mass*acceleration


def calc_gravitational_force(mass1, mass2, position1, position2):
    """
    Brechnet die Gravitionskraft, mit der eine Punktmasse mass1, die sich an der Position position1
    befindet, von einer Punktmasse mass2, die sich an der Position position2
    befindet, angezogen wird

    params:
        mass1: Punktmasse eines Koerpers 1
        mass2: Punktmasse eines zweiten Koerpers 2
        position1: Position von Koerper 1
        position2: Position von Koerper 2
    return:
        Gravitationskraft als Vektor
    """

    pos1 = np.array(position1)
    pos2 = np.array(position2)
    vector_length = np.linalg.norm(pos2-pos1)
    return G_CONSTANT*((mass1*mass2)/vector_length**3)*(pos2-pos1)

def stepwise_simulation(body, sum_forces, delta_t):
    """
    Berechnet die neue Position eines Koerpers nach einer bestimmten Zeit.

    params:
        body: Koerper fuer Berechnung
        sum_forces: Summe aller wirkenden Kraefte
        delta_t: Zeitunterschied
    """
    return body.pos + delta_t * body.speed + (delta_t**2/2)*(sum_forces/body.mass)
