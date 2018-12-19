"""
    Module to send changing object positions through a pipe.
"""
#
# Copyright (C) 2017  "Peter Roesch" <Peter.Roesch@fh-augsburg.de>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
# or open http://www.fsf.org/licensing/licenses/gpl.html
#
import sys
from random import uniform, random # TODO: C libs
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

import physics_formula as pf
import simulation_constants as sc

__FPS = 60
__DELTA_ALPHA = 0.01
cdef double G_CONSTANT = 6.673e-11


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _move_bodies_circle(double[:, ::1] positions,
                              double[:, ::1] speed,
                              double[::1] mass,
                              int timestep):
    """
    Iteriert durch alle Körper und berechnet
    ihre neue Geschwindigkeit und Position.

    params:
        positions: NumPy-Array aller Positionen der Körper
        speed: NumPy-Array aller Geschwindigkeiten der Körper
        mass: NumPy-Array aller Massen der Körper
        timestep: Anzahl der Sekunden pro berechnetem Schritt
    """

    cdef np.intp_t i, j, c
    cdef double[::1] tmp_loc = np.empty(3, dtype=np.float64)
    cdef double[::1] mass_foc_pos = np.empty(3, dtype=np.float64)
    cdef double[::1] delta_pos = np.empty(3, dtype=np.float64)
    cdef double[::1] grav_force = np.empty(3, dtype=np.float64)
    cdef double[::1] accel = np.empty(3, dtype=np.float64)
    cdef double mass_foc_weight
    cdef double tot_mass = 0.0
    cdef double accumulator = 0.0
    cdef double abs_delta_pos

    for i in range(1, mass.shape[0]):
        # TODO: np.empty isnt necessary, just a reset to (0,0,0) Tuples
        # Issue was that x, y kept being added on another without resetting every time
        tmp_loc = np.empty(3, dtype=np.float64)
        mass_foc_pos = np.empty(3, dtype=np.float64)
        delta_pos = np.empty(3, dtype=np.float64)
        grav_force = np.empty(3, dtype=np.float64)
        accel = np.empty(3, dtype=np.float64)
        tot_mass = 0.0
        accumulator = 0.0
        '''
        Idea to optimise mass focus positions calculation:
            calculate it once and then add and subtract the right
            values each time you go through the loop, instead of going
            through this current loop every time! O(mass.shape[0]**2)
        '''
        # MASS FOCUS POSITION
        for j in range(mass.shape[0]):
            tot_mass += mass[j]
            if j == i:
                continue
            for c in range(3): # range 3 because 3 dimensional vectors
                tmp_loc[c] = tmp_loc[c] + mass[j]*positions[j, c]

        mass_foc_weight = tot_mass - mass[i]
        for j in range(3):
            mass_foc_pos[j] = tmp_loc[j]/mass_foc_weight

        # G FORCE
        for j in range(3):
            delta_pos[j] = mass_foc_pos[j] - positions[i, j]
            accumulator += delta_pos[j]*delta_pos[j]
        abs_delta_pos = sqrt(accumulator)

        for j in range(3):
            # ACTUAL G FORCE CALCULATION
            grav_force[j] = G_CONSTANT * ((mass[i]/(abs_delta_pos*abs_delta_pos*abs_delta_pos))*mass_foc_weight) * delta_pos[j]
            # ACCELERATION
            accel[j] = grav_force[j] / mass[i]
            # NEXT LOCATION
            positions[i, j] = positions[i, j] + timestep * speed[i, j] + (timestep**2/2) + accel[j]
            # SPEED
            speed[i, j] = speed[i, j] + timestep * accel[j]


def _initialise_bodies(nr_of_bodies, mass_lim, dis_lim, rad_lim, black_weight):
    """
    Initialisiert eine Anzahl von Körpern mit zufälligen Massen
    und Positionen. Außerdem wird jedem Planeten eine
    Startgeschwindigkeit zugeteilt, sodass insgesamt ein
    stabiles System entsteht.

    params:
        nr_of_bodies: Anzahl der zu generierenden Planeten
    """
    min_mass = mass_lim[0]
    max_mass = mass_lim[1]
    min_radius = rad_lim[0]
    max_radius = rad_lim[1]
    min_distance = dis_lim[0]
    max_distance = dis_lim[1]
    max_z = dis_lim[2]

    black_hole_weight = black_weight

    positions = np.zeros((nr_of_bodies+1, 3), dtype=np.float64)
    speed = np.zeros((nr_of_bodies+1, 3), dtype=np.float64)
    radius = np.zeros((nr_of_bodies+1), dtype=np.float64)
    mass = np.zeros((nr_of_bodies+1), dtype=np.float64)

    # Black Hole
    positions[0] = np.array([0, 0, 0])
    speed[0] = [0, 0, 0]
    mass[0] = black_hole_weight
    radius[0] = 5000000000

    for i in range(1, nr_of_bodies+1):
        x_pos = uniform(min_distance, max_distance) * _get_sign()
        y_pos = uniform(min_distance,
                        np.sqrt(max_distance**2 - x_pos**2))*_get_sign()
        z_pos = uniform(0, max_z) * _get_sign()
        # Note: y_pos gets randomly generated between the min distance and
        #       the distance so that the length of the (x, y) vector
        #       is never longer than max_distance.
        positions[i] = np.array([x_pos,
                                 y_pos,
                                 z_pos])

        mass[i] = uniform(min_mass, max_mass)
        radius[i] = uniform(min_radius, max_radius)

    for i in range(1, nr_of_bodies+1):
        speed[i] = pf.calc_speed_direction(i, mass, positions)

    return positions, speed, radius, mass


def _get_sign():
    """
    Generiert ein zufälliges Vorzeichen -/+
    um bei der Initialisierung alle 4 Quadranten
    mit Planeten zu füllen.

    return:
        +1 / -1
    """
    return 1 if random() >= 0.5 else -1


cpdef startup(sim_pipe, nr_of_bodies, mass_lim, dis_lim, rad_lim, black_weight, timestep):
    """
        Initialise and continuously update a position list.

        Results are sent through a pipe after each update step

        Args:
            sim_pipe (multiprocessing.Pipe): Pipe to send results
            delta_t (float): Simulation step width.
    """

    positions, speed, radius, mass = _initialise_bodies(nr_of_bodies,
                                                        mass_lim,
                                                        dis_lim,
                                                        rad_lim,
                                                        black_weight)
    while True:
        if sim_pipe.poll():
            message = sim_pipe.recv()
            if isinstance(message, str) and message == sc.END_MESSAGE:
                print('simulation exiting ...')
                sys.exit(0)
        _move_bodies_circle(positions, speed, mass, timestep)
        pos_with_radius = np.c_[positions, radius]
        sim_pipe.send(pos_with_radius * (1/dis_lim[1]))
        # Positions changed in movedbodies is sent to renderer through the pipe
