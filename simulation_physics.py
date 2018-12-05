"""
    Module to send changing object positions through a pipe. Note that
    this is not a simulation, but a mockup.
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
import time
import random as rand
import numpy as np
import physics_formula as pf

import simulation_constants as sc

__FPS = 60
__DELTA_ALPHA = 0.01


def _move_bodies_circle(positions, speed, mass, delta_t):
    # This function will be responsible for setting new positions.
    timestep = delta_t*10000

    for i in range(1, mass.size):
        mass_foc_pos = pf.calc_mass_focus_ignore(i, mass, positions)
        mass_foc_weight = np.sum(mass) - mass[i]
        grav_force = pf.calc_gravitational_force(mass[i],
                                                 mass_foc_weight,
                                                 positions[i],
                                                 mass_foc_pos)
        accel = pf.calc_acceleration(grav_force, mass[i])
        speed[i] = pf.calc_speed_direction(i, mass, positions)
        positions[i] = pf.next_location(positions[i], speed[i],
                                        accel, timestep)

def _get_sign():
	return 1 if rand.random() >= 0.5 else -1

def _initialise_bodies(nr_of_bodies, mass_lim, dis_lim, rad_lim, black_weight):
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


    #Black Hole
    positions[0] = np.array([0, 0, 0])
    speed[0] = [0, 0, 0]
    mass[0] = black_hole_weight
    radius[0] = 5000000000

    for i in range(1, nr_of_bodies+1):
        positions[i] = np.array([rand.uniform(min_distance, max_distance) * _get_sign(), 
        	                     rand.uniform(min_distance, max_distance) * _get_sign(), 
        	                     rand.uniform(0, max_z) * _get_sign()])
        #TODO: as x y and z are random between -max and +max they can get too close to the black hole
        speed[i] = [0, pf.calc_absolute_speed(i, mass, positions), 0]
        mass[i] = rand.uniform(min_mass, max_mass)
        radius[i] = rand.uniform(min_radius, max_radius)


    return positions, speed, radius, mass


def startup(sim_pipe, delta_t, nr_of_bodies, mass_lim, dis_lim, rad_lim, black_weight):
    """
        Initialise and continuously update a position list.

        Results are sent through a pipe after each update step

        Args:
            sim_pipe (multiprocessing.Pipe): Pipe to send results
            delta_t (float): Simulation step width.
    """

    positions, speed, radius, mass = _initialise_bodies(nr_of_bodies, mass_lim, dis_lim, rad_lim, black_weight)
    while True:
        if sim_pipe.poll():
            message = sim_pipe.recv()
            if isinstance(message, str) and message == sc.END_MESSAGE:
                print('simulation exiting ...')
                sys.exit(0)
        _move_bodies_circle(positions, speed, mass, delta_t)
        pos_with_radius = np.c_[positions, radius]
        # print(pos_with_radius)
        sim_pipe.send(pos_with_radius * (1/dis_lim[1]))
        # Positions changed in movedbodies is sent to renderer through the pipe
