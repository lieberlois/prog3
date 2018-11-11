""" simple PyQt5 simulation controller """
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
import multiprocessing

from PyQt5 import QtWidgets
import simulation_mockup
import galaxy_renderer
from simulation_constants import END_MESSAGE

class SimulationGUI(QtWidgets.QWidget):
    """
        Widget with two buttons
    """
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setGeometry(0, 0, 260, 60)
        self.setWindowTitle('Simulation')

        self.start_button = QtWidgets.QPushButton('Start', self)
        self.start_button.setGeometry(10, 10, 60, 35)
        self.start_button.clicked.connect(self.start_simulation)

        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.setGeometry(100, 10, 60, 35)
        self.stop_button.clicked.connect(self.stop_simulation)

        self.quit_button = QtWidgets.QPushButton('Quit', self)
        self.quit_button.setGeometry(190, 10, 60, 35)
        self.quit_button.clicked.connect(self.exit_application)

        self.renderer_conn, self.simulation_conn = None, None
        self.render_process = None
        self.simulation_process = None
        multiprocessing.set_start_method('spawn')

    def start_simulation(self):
        """
            Start simulation and render process connected with a pipe.
        """
        self.renderer_conn, self.simulation_conn = multiprocessing.Pipe()
        self.simulation_process = \
            multiprocessing.Process(target=simulation_mockup.startup,
                                    args=(self.simulation_conn, 16, 1))
        self.render_process = \
            multiprocessing.Process(target=galaxy_renderer.startup,
                                    args=(self.renderer_conn, 60), )
        self.simulation_process.start()
        self.render_process.start()

    def stop_simulation(self):
        """
            Stop simulation and render process by sending END_MESSAGE
            through the pipes.
        """
        if self.simulation_process is not None:
            self.simulation_conn.send(END_MESSAGE)
            self.simulation_process = None

        if self.render_process is not None:
            self.renderer_conn.send(END_MESSAGE)
            self.render_process = None

    def exit_application(self):
        """
            Stop simulation and exit.
        """
        self.stop_simulation()
        self.close()

def _main(argv):
    """
        Main function to avoid pylint complains concerning constant names.
    """
    app = QtWidgets.QApplication(argv)
    simulation_gui = SimulationGUI()
    simulation_gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    _main(sys.argv)
