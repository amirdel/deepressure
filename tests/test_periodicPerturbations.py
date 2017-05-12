# Copyright 2017 Amir Hossein Delgoshaie, amirdel@stanford.edu
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee
# is hereby granted, provided that the above copyright notice and this permission notice appear in all
# copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
# ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from unittest import TestCase
import unittest
import os as os
import pandas as pd
from deepres.simulator.linear_system_solver import LinearSystemStandard
from deepres.simulator.periodic_field_functions import PeriodicPerturbations
import numpy as np
import pickle as pickle


class TestPeriodicPerturbations(TestCase):
    @unittest.skip("skipping")
    def test_periodic_rhs_vec(self):
        self.fail()

    @unittest.skip("skipping")
    def test_set_face_velocity(self):
        self.fail()

    def test_face_velocity_operator(self):
        proj_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # path to grid object
        grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')
        with open(grid_path, 'rb') as input:
            grid = pickle.load(input)
        # path to the permeability realizations
        perm_path = os.path.join(proj_folder, 'data', 's100_gauss01_512_5.csv')
        # mean pressure in the x directions and the y direction
        dp_x, dp_y = 1.0, 0.0
        perm_frame = pd.read_csv(perm_path, usecols=range(1))
        # initialize a linear system for the pressure fluctuations for the grid
        LS = LinearSystemStandard(grid)
        # initialize the periodic perturbation object
        PI = PeriodicPerturbations(grid, dp_x, dp_y)
        # for the number of specified realizations run particle tracking and save the results
        logperm = perm_frame.ix[:, 0]
        perm = np.exp(logperm)
        grid.set_transmissibility(perm)
        # solve for fluctuations around mean pressure gradient
        # setting the left hand side of the equation
        LS.fill_matrix(grid.transmissibility)
        # for each cell add (dp_x/lx)*(T_down - T_up)_x + (dp_y/ly)*(T_down - T_up)_y
        # to the rhs
        rhs_vec = PI.periodic_rhs_vec(grid.transmissibility)
        LS.rhs.set_neumann_pores_distributed(range(grid.nr_p), rhs_vec)
        # set a dirichlet cell: no fluctuation for cell 0
        LS.set_dirichlet_pores([0], 0.0)
        LS.solve()
        # get the face velocity by looping over faces
        face_vel_old = PI.set_face_velocity(LS.sol, grid.transmissibility)
        # get the operator and reconstruct the velocity
        face_vel_opretator, face_vel_fix_grad = PI.face_velocity_operator(grid.transmissibility)
        face_vel_new = np.dot(face_vel_opretator, LS.sol) + face_vel_fix_grad
        # print(np.max(np.abs(face_vel_new-face_vel_old)))
        np.testing.assert_almost_equal(face_vel_new, face_vel_old, decimal=10)

    @unittest.skip("skipping")
    def test_get_cell_velocity(self):
        self.fail()

    @unittest.skip("skipping")
    def test_get_total_pressure(self):
        self.fail()
