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


from PyQt5 import QtWidgets, uic
import simulation_physic
import galaxy_renderer
from simulation_constants import END_MESSAGE


class SimulationGUI(QtWidgets.QMainWindow):
    """
        Widget with Buttons and LineEdits
    """
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('Simulation')
        self.ui = uic.loadUi("simulation_gui.ui", self)

        self.startButton.clicked.connect(self.start_simulation)
        self.quitButton.clicked.connect(self.exit_application)

        self.renderer_conn, self.simulation_conn = None, None
        self.render_process = None
        self.simulation_process = None

    def start_simulation(self):
        """
            Start simulation and render process connected with a pipe.
        """
        self.stop_simulation()
        nr_of_planets = self.ui.nrPlanetSpinBox.value()
        mass_lim = (float(self.ui.minMassLineEdit.text()),\
                    float(self.ui.maxMassLineEdit.text()))
        dis_lim = (float(self.ui.minDistanceLineEdit.text())),\
                   float(self.ui.maxDistanceLineEdit.text()),\
                   float(self.ui.maxDistanceZValue.text())
        rad_lim = (float(self.ui.minRadiusLineEdit.text())), \
                   float(self.ui.maxRadiusLineEdit.text())
        black_weight = float(self.ui.blackHoleWeightLineEdit.text())

        timestep = float(self.ui.timestepValue.text())

        self.renderer_conn, self.simulation_conn = multiprocessing.Pipe()
        self.simulation_process = \
            multiprocessing.Process(target=simulation_physic.startup,
                                    args=(self.simulation_conn,
                                          nr_of_planets,
                                          mass_lim, dis_lim,
                                          rad_lim, black_weight, timestep))
        self.render_process = \
            multiprocessing.Process(target=galaxy_renderer.startup,
                                    args=(self.renderer_conn, 60), )

        # self.close()
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
    # TODO: The Compiling of our Cython files should be handled here
    _main(sys.argv)
