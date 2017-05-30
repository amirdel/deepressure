from scipy.sparse import coo_matrix

def face_velocity_operator_nonperiodic(grid):
    transRockGeometric = grid.transmissibility
    row, col, data  = [[] for i in range(3)]
    lx, ly = grid.lx, grid.ly
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    y_faces, d_cell_numbers = grid.y_faces, grid.d_cell_numbers
    face_adj_list = grid.updwn_cells
    b_counter = 0
    for face in range(grid.nr_t):
        # find adjacent cells
        adj_cells = face_adj_list[face]
        # if the face has only one adjacent cell then continue:
        if len(set(adj_cells)) == 1:
            b_counter += 1
            continue
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
    print('number of boundary cells: ', b_counter)
    return face_vel_operator

def get_cell_velocity(grid, face_velocities):
    cell_faces = grid.facelist_array
    u = 0.5 * (face_velocities[cell_faces[:, 0]] + face_velocities[cell_faces[:, 1]])
    v = 0.5 * (face_velocities[cell_faces[:, 2]] + face_velocities[cell_faces[:, 3]])
    return u,v