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

import os
import numpy as np
from deepres.simulator.grid_structured import structuredGrid
from deepres.simulator.linear_system_solver import LSGridPeriodicPurturbations

script_path = os.path.dirname(__file__)
script_files_folder = os.path.join(script_path, 'script_files')
grid_name = '100_100_periodic_100.pkl'
perm_name = 'permeability_100_100.csv'
perm_path = os.path.join(script_files_folder, perm_name)
grid_path = os.path.join(script_files_folder, grid_name)

#mean pressure gradient
dp_x = 100.0
dp_y = 0.0
#run n realizations and save results -> x,t,last_index
#define dx and dy (only supported if they are equal)
dx = 1.0
dy = 1.0
#define number of grid cells in the x direction (m) and in the y direction (n)
m = 100
n = 100
#specify boundary type -> full-periodic or non-periodic
boundaryType = 'full-periodic'
#create the grid
grid = structuredGrid(m, n, dx, dy, boundaryType=boundaryType)
#loading perm
logperm = np.loadtxt(perm_path, skiprows=1)
perm = np.exp(logperm)
# initialize a linear system for the pressure fluctuations for the grid
LS = LSGridPeriodicPurturbations(grid)
# load the permeability file and assign transmissibility to faces
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
#perform particle tracking
grid.pressure = LS.sol
grid.face_velocities = LS.set_face_velocity(dp_x, dp_y)
print(grid.face_velocities)
print('finished')