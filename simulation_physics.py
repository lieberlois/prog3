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
import math
import numpy as np
import physics_formula as pf

import simulation_constants as sc

__FPS = 60
__DELTA_ALPHA = 0.01


def _move_bodies_circle(positions, speed, mass, delta_t):  # This function will be responsible for setting new positions.
    for pos_index, pos in enumerate(positions):
        if pos_index == len(positions)-1:
            break
        force = pf.calc_gravitational_force(mass[pos_index],
                                            mass[pos_index+1],
                                            positions[pos_index],
                                            positions[pos_index+1])
        acceleration = pf.calc_acceleration(force, mass[pos_index])
        current_pos = positions[pos_index]
        #print(current_pos)
        next_loc = pf.next_location(mass[pos_index],
                                    current_pos,
                                    speed[pos_index],
                                    acceleration,
                                    delta_t)
        positions[pos_index] = next_loc
        #print(positions[pos_index])
    time.sleep(1/__FPS)


def _initialise_bodies():  # function creates our bodies. TODO: change this so different masses are set.
    body_amount = 2

    positions = np.zeros((body_amount, 3), dtype=np.float64)
    speed = np.zeros((body_amount, 3), dtype=np.float64)
    radius = np.zeros((body_amount), dtype=np.float64)
    mass = np.zeros((body_amount), dtype=np.float64)
    # first body
    positions[0][0] = 20
    positions[0][1] = 0
    positions[0][2] = 0
    radius[0] = (250*10**7)*(1/sc.AE_CONSTANT)  # size
    mass[0] = sc.EARTH_WEIGHT
    # second body
    positions[1][0] = 0
    positions[1][1] = 0
    positions[1][2] = 0
    radius[1] = (500*10**7)*(1/sc.AE_CONSTANT)  # size
    speed[1][0] = 1
    speed[1][1] = 1
    speed[1][2] = 1
    mass[1] = sc.SUN_WEIGHT

    return positions, speed, radius, mass


def startup(sim_pipe, delta_t):
    """
        Initialise and continuously update a position list.

        Results are sent through a pipe after each update step

        Args:
            sim_pipe (multiprocessing.Pipe): Pipe to send results
            delta_t (float): Simulation step width.
    """
    positions, speed, radius, mass = _initialise_bodies()
    while True:
        if sim_pipe.poll():
            message = sim_pipe.recv()
            if isinstance(message, str) and message == sc.END_MESSAGE:
                print('simulation exiting ...')
                sys.exit(0)
        _move_bodies_circle(positions, speed, mass, delta_t)
        pos_with_radius = np.c_[positions, radius]
        #print(pos_with_radius)
        sim_pipe.send(pos_with_radius)  # Positions changed in moved bodies is sent to the renderer through the pipe.
