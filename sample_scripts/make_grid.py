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

from deepres.simulator.grid_structured import structuredGrid
import pickle
import os

#This script creates a structured grid that is periodic in all directions

#define dx and dy (only supported if they are equal)
dx = 1.0
dy = 1.0
#define number of grid cells in the x direction (m) and in the y direction (n)
m = 100
n = 100
#save folder - if the folder does not exist one will be created
save_folder = '/home/amirhossein/Desktop/grid_test'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
#specify boundary type -> full-periodic or non-periodic
boundaryType = 'full-periodic'
suffix = '_periodic_100'
#save file name
grid_name = str(m) + '_'+ str(n)+ suffix +'.pkl'
#create the grid
grid = structuredGrid(m, n, dx, dy, boundaryType=boundaryType)
#save the grid
save_path = os.path.join(save_folder, grid_name)
with open(save_path,'wb') as output:
    pickle.dump(grid, output, pickle.HIGHEST_PROTOCOL)
print('Grid saved successfully.')