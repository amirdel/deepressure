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

from scipy.sparse import coo_matrix
import numpy as np

class PeriodicPerturbations(object):
    """
    Class used for solving steady state problem on periodic grids and setting mean
    pressure gradient in the x and y direction. This class contains the methods that are
    needed for the linear system and the methods for generating velocities from these periodic solutions.
    """
    def __init__(self, network, dp_x, dp_y):
        """

        :param network: a network object
        :param dp_x: pressure difference in the x directtion (dp/dx = dp_x/network.lx)
        :param dp_y: pressure difference in the y directtion (dp/dy = dp_y/network.ly)
        """
        self.network = network
        self.dp_x = dp_x
        self.dp_y = dp_y

    def periodic_rhs_vec(self, transRockGeometric):
        # TODO: needs to have at least three rows and columns, no problem for my case
        """
        find the rhs for the full periodic case
        :param transRockGeometric: face trans rock geometric
        :return rhs_vec: vector size number of cells
        """
        dp_x, dp_y = self.dp_x, self.dp_y
        grid = self.network
        lx, ly = grid.lx, grid.ly
        rhs_vec = np.zeros(grid.nr_p)
        nFaces = grid.nr_t
        faceCells = grid.updwn_cells
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        dCellNumbers, yFaces = grid.d_cell_numbers, grid.y_faces
        for face in range(nFaces):
            adj_cells = faceCells[face]
            trans = transRockGeometric[face]
            ups, dwn = adj_cells[0], adj_cells[1]
            # choose the correct component of dp for that face
            if ~yFaces[face]:
                d, dp, l = dx, dp_x, lx
            else:
                # y face
                d, dp, l = dy, dp_y, ly
            rhs_vec[ups] -= (dp/l)*trans*d
            rhs_vec[dwn] += (dp/l)*trans*d
        return rhs_vec

    def set_face_velocity(self, p_fluc, transRockGeometric):
        """
        function to set the right hand side vector when solving for pressure fluctuations
        :param transRockGeometric: face trans rock geometric
        :return: velocity of at the cell faces (grid.nr_t)
        """
        dp_x, dp_y = self.dp_x, self.dp_y
        grid = self.network
        lx, ly = grid.lx, grid.ly
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        y_faces, d_cell_numbers = grid.y_faces, grid.d_cell_numbers
        face_adj_list = grid.updwn_cells
        face_velocity = np.zeros(grid.nr_t)
        for face in range(grid.nr_t):
            # find adjacent cells
            adj_cells = face_adj_list[face]
            trans = transRockGeometric[face]
            ups, dwn = adj_cells[0], adj_cells[1]
            if ~y_faces[face]:
                d, dp, l = dx, dp_x, lx
            else:
                # y face
                d, dp, l = dy, dp_y, ly
            A = dz*d
            face_velocity[face] = trans*(d*dp/l + (p_fluc[ups]-p_fluc[dwn]))/A
        return face_velocity

    def face_velocity_operator(self, transRockGeometric):
        """

        :param transRockGeometric:
        :return: A, b s.t np.dot(A, p_fluc) + b = face_velocities
        """
        row, col, data  = [[] for i in range(3)]
        dp_x, dp_y = self.dp_x, self.dp_y
        grid = self.network
        face_vel_fix_grad = np.zeros(grid.nr_t)
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
                d, dp, l = dx, dp_x, lx
            else:
                # y face
                d, dp, l = dy, dp_y, ly
            A = dz * d
            # face_vel_opretator[face, ups] = trans/A
            row.append(face)
            col.append(ups)
            data.append(trans/A)
            # face_vel_opretator[face, dwn] = -trans/A
            row.append(face)
            col.append(dwn)
            data.append(-trans / A)
            face_vel_fix_grad[face] = trans*(d*dp/l)/A
        face_vel_operator = coo_matrix((data, (row, col)), shape=(grid.nr_t, grid.nr_p)).tocsc()
        return face_vel_operator, face_vel_fix_grad

    def get_cell_velocity(self):
        grid = self.network
        face_velocities = grid.face_velocities
        cell_faces = grid.facelist_array
        u = 0.5*(face_velocities[cell_faces[:,0]] + face_velocities[cell_faces[:,1]])
        v = 0.5 * (face_velocities[cell_faces[:, 2]] + face_velocities[cell_faces[:, 3]])
        return u,v

    def get_total_pressure(self, dp_x, dp_y):
        pass