"""
Modul zur physikalischen Berechnung von Kraeften
"""
import numpy as np
from simulation_constants import G_CONSTANT


def calc_acceleration(force, mass):
    """
    Funktion zur Berechnung der Beschleunigung, aus der Formel F = m*a

    params:
        force: Kraft nach Newton
        mass: Masse des Koerpers
    return:
        Beschleunigung
    """
    if mass <= 0:
        raise TypeError('Mass has to be greater than 0')
    return (1/mass)*force


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
    #print(position1)
    delta_pos = np.linalg.norm(pos2-pos1)
    return G_CONSTANT*((mass1*mass2)/delta_pos**3)*(pos2-pos1)


def next_location(mass, position, speed, acceleration, delta_t):
    """
    Berechnet die neue Position eines Koerpers nach einer bestimmten Zeit.

    params:
        mass: Masse des Koerpers
        position: Position des Koerpers
        speed: Geschwindigkeit des Koerpers
        sum_forces: Summe aller wirkenden Kraefte
        delta_t: Zeitunterschied
    """
    return position + delta_t * speed + (delta_t**2/2)*acceleration
