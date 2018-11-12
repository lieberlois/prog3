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

from simulation_constants import END_MESSAGE, AE_CONSTANT

__FPS = 60
__DELTA_ALPHA = 0.01


def _move_bodies(bodies, delta_t):  # This function will be responsible for setting new positions.
    for body_index, body in enumerate(bodies):
        j = len(bodies) - body_index
        sin_a = math.sin(__DELTA_ALPHA * j * delta_t * 0.2) # 0.2 slows this down by 1/5
        cos_a = math.cos(__DELTA_ALPHA * j * delta_t * 0.2)
        pos_x = body[0]
        pos_y = body[1]
        body[0] = pos_x * cos_a - pos_y * sin_a
        body[1] = pos_x * sin_a + pos_y * cos_a
    time.sleep(1/__FPS)


def _initialise_bodies(nr_of_bodies):  # function creates our bodies. TODO: change this so different masses are set.
    body_array = np.zeros((nr_of_bodies, 4), dtype=np.float64)
    body_array[0][0] = 0
    body_array[0][1] = 0
    body_array[0][2] = 0
    body_array[0][3] = (500*10**7)*(1/AE_CONSTANT) #size
    #body_array[0][4] = 0 #speed
    #body_array[0][5] = 1.989*10**30 #mass
    body_array[1][0] = 1
    body_array[1][1] = 0
    body_array[1][2] = 0
    body_array[1][3] = (250*10**7)*(1/AE_CONSTANT) #size
    #body_array[1][4] = 29780 #speed of earth
    #body_array[1][5] = 5.972 * 10 ** 24 #mass

    return body_array


def startup(sim_pipe, nr_of_bodies, delta_t):
    """
        Initialise and continuously update a position list.

        Results are sent through a pipe after each update step

        Args:
            sim_pipe (multiprocessing.Pipe): Pipe to send results
            nr_of_bodies (int): Number of bodies to be created and updated.
            delta_t (float): Simulation step width.
    """
    bodies = _initialise_bodies(nr_of_bodies)
    while True:
        if sim_pipe.poll():
            message = sim_pipe.recv()
            if isinstance(message, str) and message == END_MESSAGE:
                print('simulation exiting ...')
                sys.exit(0)
        _move_bodies(bodies, delta_t)
        sim_pipe.send(bodies)  # Positions changed in moved bodies is sent to the renderer through the pipe.
