import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def side_by_side(x_mat, y_mat, mat1, mat2, save_path):
    fig = plt.figure(figsize=(9, 4))

    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.1, right=0.86, hspace=0.1)
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.89, right=0.92)
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax_cmap = plt.subplot(gs1[0])

    # cmap = plt.cm.afmhot_r
    cmap = plt.cm.coolwarm
    vmin = min(np.amin(mat1), np.amin(mat2))
    vmax = max(np.amax(mat1), np.amax(mat2))
    p = ax.pcolormesh(x_mat, y_mat, mat1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal', 'box')
    p2 = ax2.pcolormesh(x_mat, y_mat, mat2, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(p2, ax_cmap, orientation='vertical')
    ax.set_aspect('equal', 'box')
    fig.savefig(save_path, format='png')
    plt.close(fig)
