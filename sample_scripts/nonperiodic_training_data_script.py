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

import os as os
from deepres.simulator.generate_training_data import generate_continuum_realizations_nonperiodic

proj_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# path to save the training data file
save_folder = os.path.join(proj_folder, 'temp')
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
save_path = os.path.join(save_folder, 'data_nonperiodic.npz')
# path to grid object
grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')
# path to the permeability realizations
perm_path = os.path.join(proj_folder, 'data', 's100_gauss01_512_5.csv')
# mean pressure in the x directions and the y direction
dp_x, dp_y = 1.0, 0.0
# number of training images to create
n_images = 10
generate_continuum_realizations_nonperiodic(grid_path, save_path, perm_path, n_images)