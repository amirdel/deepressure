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
from deepres.simulator.linear_system_solver import LinearSystemStandard
from deepres.simulator.periodic_field_functions import PeriodicPerturbations
from deepres.simulator.operators import face_velocity_operator_nonperiodic

def generate_continuum_realizations_periodic(grid_path, save_path, perm_path, dp_x, dp_y, n_images, print_every=50):
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
    U_face = np.zeros((n_images, n_face))
    # initialize the array for saving the face operator and bias
    face_operator_list = []
    face_bias_array = np.zeros((n_images, n_face))
    # load the permeability dataframe, each column is one realization
    # this is the file saved by SGEMS (Geostats software)
    perm_frame = pd.read_csv(perm_path, usecols=range(n_images))
    # initialize a linear system for the pressure fluctuations for the grid
    LS = LinearSystemStandard(grid)
    # initialize the perturbation system object
    PI = PeriodicPerturbations(grid, dp_x, dp_y)
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
        rhs_vec = PI.periodic_rhs_vec(grid.transmissibility)
        LS.rhs.set_neumann_pores_distributed(range(grid.nr_p), rhs_vec)
        # set a dirichlet cell: no fluctuation for cell 0
        LS.set_dirichlet_pores([0], 0.0)
        LS.solve()
        # copy the pressure solution and the permeability field to the X and Y
        X[i, :, :, 0] = np.reshape(logperm, (n_cell, n_cell))
        Y[i, :, :, 0] = np.copy(np.reshape(LS.sol, (n_cell, n_cell)))
        grid.pressure = LS.sol
        # get the operators to calculate face velocity
        U_face_operator, U_face_fixed = PI.face_velocity_operator(grid.transmissibility)
        # save face_velocity
        U_face[i,:] = U_face_operator.dot(LS.sol) + U_face_fixed
        # save the face operator
        face_operator_list.append(U_face_operator)
        face_bias_array[i,:] = U_face_fixed
    # save X, Y, U_face, operators
    np.savez(save_path, X=X, Y=Y, U_face=U_face, U_face_operator=face_operator_list, U_face_fixed=face_bias_array)


def generate_continuum_realizations_nonperiodic(grid_path, save_path, perm_path, n_images, print_every=50):
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
    # find the indices for the left and right boundary
    cell_x = grid.pores.x
    x_min, x_max = np.amin(cell_x), np.amax(cell_x)
    idxmin = np.where(cell_x == x_min)[0]
    idxmax = np.where(cell_x == x_max)[0]
    # initialize perm matrix and pressure solution (n_cell x n_cell x n_perm_fields)
    X, Y = [np.zeros((n_images, n_cell, n_cell, 1)) for i in range(2)]
    # initialize arrays for saving the faces velocities
    U_face = np.zeros((n_images, n_face))
    # initialize the array for saving the face operator and bias
    face_operator_list = []
    face_bias_array = np.zeros((n_images, n_face))
    # load the permeability dataframe, each column is one realization
    # this is the file saved by SGEMS (Geostats software)
    perm_frame = pd.read_csv(perm_path, usecols=range(n_images))
    # initialize a linear system for the pressure fluctuations for the grid
    LS = LinearSystemStandard(grid)
    # for the number of specified realizations run particle tracking and save the results
    for i in range(n_images):
        if not i%print_every:
            print('realization number '+str(i))
        logperm = perm_frame.ix[:, i]
        perm = np.exp(logperm)
        grid.set_transmissibility(perm)
        # setting the left hand side of the equation
        LS.fill_matrix(grid.transmissibility)
        # set dirichlet BC
        LS.set_dirichlet_pores(idxmin, 1.0)
        LS.set_dirichlet_pores(idxmax, 0.0)
        # solve for pressure
        LS.solve()
        # copy the pressure solution and the permeability field to the X and Y
        X[i, :, :, 0] = np.reshape(logperm, (n_cell, n_cell))
        Y[i, :, :, 0] = np.copy(np.reshape(LS.sol, (n_cell, n_cell)))
        grid.pressure = LS.sol
        # get the operators to calculate face velocity
        # U_face_operator = face_velocity_operator(grid.transmissibility)
        U_face_operator = face_velocity_operator_nonperiodic(grid)
        # save face_velocity
        U_face[i,:] = U_face_operator.dot(LS.sol)
        # U_face[i, :] = np.zeros(n_face)
        # save the face operator
        face_operator_list.append(U_face_operator)
    # save X, Y, U_face, operators
    np.savez(save_path, X=X, Y=Y, U_face=U_face, U_face_operator=face_operator_list)
