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


def calc_gravitational_force(mass1, mass2, pos1, pos2):
    """
    Brechnet Gravitionskraft, mit der eine Punktmasse mass1, die sich an pos1
    befindet, von einer Punktmasse mass2, die sich an pos2
    befindet, angezogen wird

    params:
        mass1: Punktmasse eines Koerpers 1
        mass2: Punktmasse eines zweiten Koerpers 2
        pos1: Position von Koerper 1 als numpy Array
        pos2: Position von Koerper 2 als numpy Array
    return:
        Gravitationskraft als Vektor
    """

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
    return (position + delta_t * speed + (delta_t**2/2)*acceleration)
