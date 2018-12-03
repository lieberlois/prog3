"""
Modul zur physikalischen Berechnung von Kraeften
"""

import math
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

    delta_pos = np.linalg.norm(pos2 - pos1)
    return G_CONSTANT * (((mass1)/delta_pos**3)*mass2) * (pos2 - pos1)


def next_location(position, speed, acceleration, delta_t):
    """
    Berechnet die neue Position eines Koerpers nach einer bestimmten Zeit.

    params:
        position: Position des Koerpers
        speed: Geschwindigkeit des Koerpers
        sum_forces: Summe aller wirkenden Kraefte
        delta_t: Zeitunterschied
    """
    return position + delta_t * speed + (delta_t**2/2)*acceleration


def total_mass(masses):
    """
    Berechnet die Gesamtmasse M aller Körper

    params:
        masses: Liste aller Massen
    """
    return np.sum(masses)


def calc_mass_focus(masses, positions):
    """
    Berechnet die Position des Massen-Schwerpunkts

    params:
        masses: Liste aller Massen
        positions: Liste aller Positions
    """
    tmp_focus = np.zeros(3, dtype=np.float64)
    for i in range(masses.size):
        tmp_focus = tmp_focus + masses[i] * positions[i]
    return (1/total_mass(masses)) * tmp_focus


def calc_mass_focus_ignore(ignore, masses, positions):
    """
    Berechnet die Position des Massenfokuspunktes im Raum.

    params:
        ignore: Index des zu ignorierenden Planets
        masses: Liste aller Massen
        positions: Liste aller Positionen
    """
    tmp_loc = np.zeros(3, dtype=np.float64)

    for i in range(masses.size):
        if i == ignore:
            continue
        tmp_loc = tmp_loc + (masses[i] * positions[i])
    return (1/(total_mass(masses) - masses[i]))*tmp_loc


def calc_momentum(masses, speeds):
    """
    Berechnet Gesamtimpuls des Systems

    params:
        masses: Liste aller Massen
        speeds: Liste aller Geschwindigkeiten
    """
    tmp_momentum = np.zeros(3, dtype=np.float64)
    for i in range(masses.size):
        tmp_momentum = tmp_momentum + masses[i]*speeds[i]
    return tmp_momentum


def calc_absolute_speed(body_index, masses, positions):
    """
    Berechnet den Betrag der Geschwindigkeit für den Körper body_index

    params:
        body_index: Index des gewünschten Körpers
        masses: Liste aller Massen
        positions: Liste aller Positionen
    """
    my_mass, my_position = masses[body_index], positions[body_index]
    mass_focus_ignored = calc_mass_focus_ignore(body_index, masses, positions)

    r_vector = np.linalg.norm(my_position - mass_focus_ignored)
    return ((total_mass(masses) - my_mass) /
            total_mass(masses))*math.sqrt(G_CONSTANT*total_mass(masses)/r_vector)


def calc_speed_direction(body_index, masses, positions):
    """
    Berechnet die Richtung der Geschwindigkeit für den Körper body_index

    params:
        body_index: Index des gewünschten Körpers
        masses: Liste aller Massen
        positions: Liste aller Positionen
    """
    my_absolute_speed = calc_absolute_speed(body_index, masses, positions)
    my_position = positions[body_index]
    mass_focus_ignored = calc_mass_focus_ignore(body_index, masses, positions)

    z_vector = np.array([0, 0, 1])
    cross_product = np.cross((my_position-mass_focus_ignored), z_vector)
    return cross_product / np.linalg.norm(cross_product) * my_absolute_speed
