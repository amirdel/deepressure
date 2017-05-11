# Copyright 2017 Amir Hossein Delgoshaie, Zhi Yang Wong, amirdel@stanford.edu, zhiyangw@stanford.edu
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

import numpy as np
import pickle as pickle
import pandas as pd
from deepres.simulator.linear_system_solver import LSGridPeriodicPurturbations

def generate_continuum_realizations(grid_path, save_path, perm_path, dp_x, dp_y, n_images, print_every=50):
    """
    This function will create the training data for our model
    :param grid_path: full path to a grid object (../grid.pkl)
    :param save_path: full path for saveing the (../*.npz)
    :param perm_path: full path to permeability realiztions (.../*.csv)
    :param dp_x: mean pressure gradient in the x direction
    :param dp_y: mean pressure gradient in the y direction
    :param n_images: number of images/training data
    :param print_every: print every for output messages
    :return:
    """
    # loading the grid
    with open(grid_path, 'rb') as input:
        grid = pickle.load(input)
    n_cell = grid.m
    n_face = grid.nr_t
    # initialize perm matrix and pressure solution (n_cell x n_cell x n_perm_fields)
    X, Y = [np.zeros((n_images, n_cell, n_cell, 1)) for i in range(2)]
    # initialize arrays for saving the faces velocities
    U_face = np.zeros((n_face, n_images))
    # load the permeability dataframe, each column is one realization
    # this is the file saved by SGEMS (Geostats software)
    perm_frame = pd.read_csv(perm_path, usecols=range(n_images))
    # initialize a linear system for the pressure fluctuations for the grid
    LS = LSGridPeriodicPurturbations(grid)
    # for the number of specified realizations run particle tracking and save the results
    for i in range(n_images):
        if not i%print_every:
            print('realization number '+str(i))
        logperm = perm_frame.ix[:, i]
        perm = np.exp(logperm)
        grid.set_transmissibility(perm)
        # solve for fluctuations around mean pressure gradient
        # setting the left hand side of the equation
        LS.fill_matrix(grid.transmissibility)
        # for each cell add (dp_x/lx)*(T_down - T_up)_x + (dp_y/ly)*(T_down - T_up)_y
        # to the rhs
        rhs_vec = LS.periodic_rhs_vec(dp_x, dp_y)
        LS.rhs.set_neumann_pores_distributed(range(grid.nr_p), rhs_vec)
        # set a dirichlet cell: no fluctuation for cell 0
        LS.set_dirichlet_pores([0], 0.0)
        LS.solve()
        # copy the pressure solution and the permeability field to the X and Y
        X[i, :, :, 0] = np.reshape(logperm, (n_cell, n_cell))
        Y[i, :, :, 0] = np.copy(np.reshape(LS.sol, (n_cell, n_cell)))
        # perform particle tracking
        grid.pressure = LS.sol
        grid.face_velocities = LS.set_face_velocity(dp_x, dp_y)
        # u, v = LS.get_cell_velocity()
        # save u and v
        U_face[:,i] = grid.face_velocities[:]
    # save X, Y, U, V
    np.savez(save_path, X=X, Y=Y, U_face=U_face)
