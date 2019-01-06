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
from random import uniform # TODO: C libs
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from libc.stdlib cimport rand, RAND_MAX, srand

import physics_formula as pf
import simulation_constants as sc

cdef int __FPS = 60
cdef double __DELTA_ALPHA = 0.01
cdef double G_CONSTANT = 6.673e-11


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _move_bodies_circle(double[:, ::1] positions,
                              double[:, ::1] speed,
                              double[::1] mass,
                              double timestep):
    """
    Iteriert durch alle Körper und berechnet
    ihre neue Geschwindigkeit und Position.

    params:
        positions: NumPy-Array aller Positionen der Körper
        speed: NumPy-Array aller Geschwindigkeiten der Körper
        mass: NumPy-Array aller Massen der Körper
        timestep: Anzahl der Sekunden pro berechnetem Schritt
    """

    cdef np.intp_t i, j
    cdef double[::1] tmp_loc = np.empty(3, dtype=np.float64)
    cdef double[::1] delta_pos = np.empty(3, dtype=np.float64)
    cdef double[::1] accel = np.empty(3, dtype=np.float64)
    cdef double mass_foc_weight
    cdef double accumulator = 0.0
    cdef double abs_delta_pos

    for i in range(1, mass.shape[0]):
        '''
        Idea to optimise mass focus positions calculation
        and mass focus weight calculation:
            calculate it once and then add and subtract the right
            values each time you go through the loop, instead of going
            through this current loop every time! O(mass.shape[0]**2)
        '''
        # MASS FOCUS POSITION
        mass_foc_weight = 0.0
        tmp_loc = np.zeros(3, dtype=np.float64)
        for j in range(mass.shape[0]):
            if j == i:
                continue
            mass_foc_weight += mass[j]

            tmp_loc[0] = tmp_loc[0] + mass[j] * positions[j][0]
            tmp_loc[1] = tmp_loc[1] + mass[j] * positions[j][1]
            tmp_loc[2] = tmp_loc[2] + mass[j] * positions[j][2]

        accumulator = 0.0
        for j in range(3):
            tmp_loc[j] = tmp_loc[j]/mass_foc_weight

            delta_pos[j] = tmp_loc[j] - positions[i, j]

            accumulator += delta_pos[j]*delta_pos[j]

        abs_delta_pos = sqrt(accumulator)

        for j in range(3):
            # G FORCE TO ACCELERATION
            accel[j] = G_CONSTANT * (mass_foc_weight/(abs_delta_pos*abs_delta_pos*abs_delta_pos)) * delta_pos[j]
            # NEXT LOCATION
            positions[i][j] = positions[i][j] + timestep * speed[i][j] + (timestep*timestep/2.0) * accel[j]
            # SPEED
            speed[i][j] = speed[i][j] + timestep * accel[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _initialise_bodies(int nr_of_bodies, mass_lim, dis_lim, rad_lim, double black_weight):
    """
    Initialisiert eine Anzahl von Körpern mit zufälligen Massen
    und Positionen. Außerdem wird jedem Planeten eine
    Startgeschwindigkeit zugeteilt, sodass insgesamt ein
    stabiles System entsteht.

    params:
        nr_of_bodies: Anzahl der zu generierenden Planeten
    """
    cdef double min_mass = mass_lim[0]
    cdef double max_mass = mass_lim[1]
    cdef double min_radius = rad_lim[0]
    cdef double max_radius = rad_lim[1]
    cdef double min_distance = dis_lim[0]
    cdef double max_distance = dis_lim[1]
    cdef double max_z = dis_lim[2]

    black_hole_weight = black_weight

    cdef double[:, ::1] positions = np.zeros((nr_of_bodies+1, 3), dtype=np.float64)
    cdef double[:, ::1] speed = np.zeros((nr_of_bodies+1, 3), dtype=np.float64)
    
    cdef double[::1] radius = np.zeros((nr_of_bodies+1), dtype=np.float64)
    cdef double[::1] mass = np.zeros((nr_of_bodies+1), dtype=np.float64)
    
    cdef double[::1] tmp_speed = np.zeros((nr_of_bodies+1), dtype=np.float64)

    ### VARIABLES NEEDED FOR THE CALCULATION OF ABSOLUTE SPEED
    cdef double tot_mass_ignored = 0.0
    cdef double tot_mass = 0.0
    cdef double accumulator = 0.0
    cdef double abs_delta
    cdef double[::1] tmp_loc = np.empty(3, dtype=np.float64)
    cdef double[::1] delta_pos = np.empty(3, dtype=np.float64)
    cdef double abs_speed

    ### VARIABLES NEEDED FOR THE CALCULATION OF SPEED DIRECTION
    cdef int[::1] z_vector = np.array([0, 0, 1], dtype=np.int32)
    cdef double[::1] cross_product = np.empty(3, dtype=np.float64)



    # Black Hole
    positions[0][0] = 0
    positions[0][1] = 0
    positions[0][2] = 0
    speed[0][0] = 0
    speed[0][1] = 0
    speed[0][2] = 0
    mass[0] = black_hole_weight
    radius[0] = 5000000000
    
    cdef int i, j

    print("generating bodies...")

    for i in range(1, nr_of_bodies+1):
        #TODO: This always produces the same numbers!
        #Eventuelle Lösung: Selbst geschriebener Random-Generator
        #                   aus Einzelabgabe!
        #x_pos = (min_distance + rand()/(RAND_MAX*max_distance))*_get_sign()
        #y_pos = (min_distance + rand()/(RAND_MAX*sqrt(max_distance**2 - x_pos**2)))*_get_sign()
        #z_pos = (min_distance + rand()/RAND_MAX*max_z)*_get_sign()
        
        x_pos = uniform(min_distance, max_distance) * _get_sign()
        y_pos = uniform(min_distance,
                        np.sqrt(max_distance**2 - x_pos**2))*_get_sign()
        z_pos = uniform(0, max_z) * _get_sign()
        # Note: y_pos gets randomly generated between the min distance and
        #       the distance so that the length of the (x, y) vector
        #       is never longer than max_distance.

        positions[i][0] = x_pos
        positions[i][1] = y_pos
        positions[i][2] = z_pos

        #TODO: This always produces the same numbers!
        #mass[i] = min_mass + rand()/(RAND_MAX*max_mass)
        #radius[i] = min_radius + rand()/(RAND_MAX*max_radius)
        
        mass[i] = uniform(min_mass, max_mass)
        radius[i] = uniform(min_radius, max_radius)


    print("calculating starting speeds...")

    # CALCULATING TOTAL MASS
    for i in range(nr_of_bodies):
        tot_mass += mass[i]

    for i in range(1, nr_of_bodies+1):
        # ABSOLUTE SPEED CALCULATION STARTS HERE
        tot_mass_ignored = 0.0
        tmp_loc[0] = 0.0
        tmp_loc[1] = 0.0
        tmp_loc[2] = 0.0
        for j in range(nr_of_bodies+1):
            if j == i:
                continue
            tot_mass_ignored += mass[j]
            tmp_loc[0] = tmp_loc[0] + mass[j] * positions[j][0]
            tmp_loc[1] = tmp_loc[1] + mass[j] * positions[j][1]
            tmp_loc[2] = tmp_loc[2] + mass[j] * positions[j][2]

        accumulator = 0.0
        for j in range(3):
            tmp_loc[j] = tmp_loc[j] / tot_mass_ignored
            delta_pos[j] = positions[i][j] - tmp_loc[j]
            accumulator += delta_pos[j] * delta_pos[j]
        abs_delta = sqrt(accumulator)
        abs_speed = (tot_mass_ignored/tot_mass)*sqrt(G_CONSTANT*tot_mass/abs_delta)
        # ABSOLUTE SPEED CALCULATION ENDS HERE

        # SPEED DIRECTION CALCULATION STARTS HERE
        cross_product = np.cross(delta_pos, z_vector)
        accumulator = 0.0
        for j in range(3):
            accumulator += cross_product[j]*cross_product[j]
        accumulator = sqrt(accumulator)
        # SPEED DIRECTION CALCULATION ENDS HERE

        speed[i][0] = cross_product[0] / accumulator * abs_speed
        speed[i][1] = cross_product[1] / accumulator * abs_speed
        speed[i][2] = cross_product[2] / accumulator * abs_speed


    return positions, speed, radius, mass


    
    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _get_sign():
    """
    Generiert ein zufälliges Vorzeichen -/+
    um bei der Initialisierung alle 4 Quadranten
    mit Planeten zu füllen.

    return:
        +1 / -1
    """
    return 1 if (<double>rand()/<double>RAND_MAX) >= 0.5 else -1


cpdef startup(sim_pipe, nr_of_bodies, mass_lim, dis_lim, rad_lim, black_weight, timestep):
    """
        Initialise and continuously update a position list.

        Results are sent through a pipe after each update step

        Args:
            sim_pipe (multiprocessing.Pipe): Pipe to send results
            delta_t (float): Simulation step width.
    """

    cdef double[:, ::1] positions = np.empty((nr_of_bodies+1, 3), dtype=np.float64)
    cdef double[:, ::1] speed = np.empty((nr_of_bodies+1, 3), dtype=np.float64)
    cdef double[::1] radius = np.empty(nr_of_bodies+1, dtype=np.float64)
    cdef double[::1] mass = np.empty(nr_of_bodies+1, dtype=np.float64)

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
