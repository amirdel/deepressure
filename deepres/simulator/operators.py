from scipy.sparse import coo_matrix
import numpy as np

def face_velocity_operator_nonperiodic(grid):
    transRockGeometric = grid.transmissibility
    row, col, data  = [[] for i in range(3)]
    lx, ly = grid.lx, grid.ly
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    y_faces, d_cell_numbers = grid.y_faces, grid.d_cell_numbers
    face_adj_list = grid.updwn_cells
    for face in range(grid.nr_t):
        # find adjacent cells
        adj_cells = face_adj_list[face]
        trans = transRockGeometric[face]
        ups, dwn = adj_cells[0], adj_cells[1]
        if ~y_faces[face]:
            d, l = dx, lx
        else:
            # y face
            d, l = dy, ly
        A = dz * d
        # face_vel_opretator[face, ups] = trans/A
        row.append(face)
        col.append(ups)
        data.append(trans/A)
        # face_vel_opretator[face, dwn] = -trans/A
        row.append(face)
        col.append(dwn)
        data.append(-trans / A)
    face_vel_operator = coo_matrix((data, (row, col)), shape=(grid.nr_t, grid.nr_p))
    return face_vel_operator

def get_cell_velocity(grid, face_velocities):
    cell_faces = grid.facelist_array
    u = 0.5 * (face_velocities[cell_faces[:, 0]] + face_velocities[cell_faces[:, 1]])
    v = 0.5 * (face_velocities[cell_faces[:, 2]] + face_velocities[cell_faces[:, 3]])
    return u,v

def divergence_operator(grid):
    row, col, data  = [np.empty([0]) for i in range(3)]
    # order of ngh_faces : left, right, bottom , top, -1 if missing
    ngh_faces = grid.facelist_array
    n_cells = grid.nr_p
    for cell in range(n_cells):
        cell_faces = ngh_faces[cell]
        mask = (cell_faces != -1)
        sgn = np.array([-1.0, 1.0, -1.0, 1.0])
        current_val = sgn[mask]
        current_row = cell* np.ones(4, dtype=np.int)
        row = np.hstack((row, current_row[mask]))
        col = np.hstack((col, cell_faces[mask]))
        data = np.hstack((data, current_val))
    return coo_matrix((data[1:], (row[1:], col[1:])), shape=(grid.nr_p, grid.nr_t))
