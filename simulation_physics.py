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
# MERCHANTABILITY or FITfrom numba import jitNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
# or open http://www.fsf.org/licensing/licenses/gpl.html
#
import sys
from random import uniform, random
import numpy as np
import physics_formula as pf
import simulation_constants as sc

__FPS = 60
__DELTA_ALPHA = 0.01

def _move_bodies_circle(positions, speed, mass, timestep):
    # This function will be responsible for setting new positions.
    for i in range(mass.size):
        mass_foc_pos = pf.calc_mass_focus_ignore(i, mass, positions)
        mass_foc_weight = np.sum(mass) - mass[i]
        grav_force = pf.calc_gravitational_force(mass[i],
                                                 mass_foc_weight,
                                                 positions[i],
                                                 mass_foc_pos)
        accel = pf.calc_acceleration(grav_force, mass[i])
        speed[i] = speed[i] + timestep * accel
        positions[i] = pf.next_location(positions[i], speed[i],
                                        accel, timestep)

def _get_sign():
    return 1 if random() >= 0.5 else -1

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

    # Black Hole
    positions[0] = np.array([0, 0, 0])
    speed[0] = [0, 0, 0]
    mass[0] = black_hole_weight
    radius[0] = 5000000000

    for i in range(1, nr_of_bodies+1):
        x_pos = uniform(min_distance, max_distance) * _get_sign()
        y_pos = uniform(min_distance,
                        np.sqrt(max_distance**2 - x_pos**2))*_get_sign()
        # Note: y_pos gets randomly generated between the min distance and
        #       the distance so that the length of the (x, y) vector
        #       is never longer than max_distance.
        positions[i] = np.array([x_pos,
                                 y_pos,
                                 uniform(0, max_z) * _get_sign()])

        mass[i] = uniform(min_mass, max_mass)
        radius[i] = uniform(min_radius, max_radius)

    for i in range(1, nr_of_bodies+1):
        speed[i] = pf.calc_speed_direction(i, mass, positions)

    return positions, speed, radius, mass


def _get_sign():
    return 1 if random() >= 0.5 else -1


def startup(sim_pipe, nr_of_bodies, mass_lim, dis_lim, rad_lim, black_weight, timestep):
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
