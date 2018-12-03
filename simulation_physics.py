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
import numpy as np
import physics_formula as pf
import random as rand

import simulation_constants as sc

__FPS = 60
__DELTA_ALPHA = 0.01


def _move_bodies_circle(positions, speed, mass, delta_t):
    # This function will be responsible for setting new positions.
    timestep = delta_t*60*24

    for i in range(mass.size):
        mass_foc_pos = pf.calc_mass_focus_ignore(i, mass, positions)
        mass_foc_weight = np.sum(mass) - mass[i]
        grav_force = pf.calc_gravitational_force(mass[i],
                                                 mass_foc_weight,
                                                 positions[i],
                                                 mass_foc_pos)
        accel = pf.calc_acceleration(grav_force, mass[i])
        speed[i] = pf.calc_speed_direction(i, mass, positions)
        positions[i] = pf.next_location(mass[i], positions[i], speed[i],
                                        accel, timestep)

    time.sleep(1/__FPS)


def _initialise_bodies(nr_of_bodies):
    # TODO: initialise bodies based on nr_of_bodies

    min_mass = 1.00*10**15
    max_mass = 1.00*10**27
    min_radius = 1000000000
    max_radius = 4000000000
    min_distance = 10000000000
    max_distance = sc.AE_CONSTANT #TODO: UI!
    
    black_hole_weight = 2.00*10**30

    positions = np.zeros((nr_of_bodies+1, 3), dtype=np.float64)
    speed = np.zeros((nr_of_bodies+1, 3), dtype=np.float64)
    radius = np.zeros((nr_of_bodies+1), dtype=np.float64)
    mass = np.zeros((nr_of_bodies+1), dtype=np.float64)

    #Black Hole
    positions[0] = np.array([0, 0, 0])
    speed[0] = [0, 0, 0]
    mass[0] = 2.00*10**30
    radius[0] = 5000000000

    for i in range(1,nr_of_bodies+1):
        #Black Hole
        positions[i] = np.array([rand.uniform(1.0*10**10, max_distance), 0, 0])
        speed[i] = [0, pf.calc_absolute_speed(i, mass, positions), 0]
        mass[i] = rand.uniform(min_mass, max_mass)
        radius[i] = rand.uniform(min_radius, max_radius)	


    return positions, speed, radius, mass


def startup(sim_pipe, delta_t, nr_of_bodies):
    """
        Initialise and continuously update a position list.

        Results are sent through a pipe after each update step

        Args:
            sim_pipe (multiprocessing.Pipe): Pipe to send results
            delta_t (float): Simulation step width.
    """

    max_distance = sc.AE_CONSTANT #TODO: UI!

    positions, speed, radius, mass = _initialise_bodies(nr_of_bodies)
    while True:
        if sim_pipe.poll():
            message = sim_pipe.recv()
            if isinstance(message, str) and message == sc.END_MESSAGE:
                print('simulation exiting ...')
                sys.exit(0)
        _move_bodies_circle(positions, speed, mass, delta_t)
        pos_with_radius = np.c_[positions, radius]
        # print(pos_with_radius)
        sim_pipe.send(pos_with_radius * (1/(max_distance)))
        # Positions changed in movedbodies is sent to renderer through the pipe
