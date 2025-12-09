# -*- coding: utf-8 -*-
"""
Created by Alfredo Camara Casado
For extra capabilities of the code, code development, and bug reporting please contact the author at: acamara@ciccp.es
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.linalg
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy import sparse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import newton

class FEModel:
    def __init__(self):
        self.nodes = {}
        self.elements = {}
        self.element_sets = {}
        self.K_global = None
        self.K_geo_global = None
        self.M_global = None
        self.dof = 6
        self.node_index_map = {}
        self.script_dir = os.getcwd()
        self.links = []
        self.prestress_forces = {}
        self.gravity_direction = np.array([0, 0, -1])  # Default: downward in Z
        self.g = 9.81  # Gravity acceleration (m/s²)


        # Element type handlers
        self.element_handlers = {
            '3D_BEAM': self._handle_beam_element,
            'TRUSS': self._handle_truss_element
            }

    def load_from_input_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.split('#')[0].strip()  # Remove inline comments
            if not line:
                continue

            parts = line.split()
            keyword = parts[0].upper()
            args = parts[1:]

            if keyword == 'NODE':
                node_id = int(args[0])
                coords = tuple(map(float, args[1:4]))
                self.nodes[node_id] = np.array(coords)



            elif keyword == 'NGEN':
                start_id = int(args[0])
                end_id = int(args[1])
                id_increment = int(args[2])

                start_coords = self.nodes[start_id]
                end_coords = self.nodes[end_id]
                total_length = np.linalg.norm(end_coords - start_coords)

                num_nodes = ((end_id - start_id) // id_increment) + 1
                if num_nodes < 2:
                    raise ValueError("Invalid node range or increment.")

                for i in range(1, num_nodes - 1):  # Exclude start and end nodes
                    new_id = start_id + i * id_increment
                    interp = tuple(
                        start_coords[j] + (end_coords[j] - start_coords[j]) * i / (num_nodes - 1)
                        for j in range(3)
                    )
                    self.nodes[new_id] = np.array(interp)


            elif keyword == 'ELEMENT':
                elem_id = int(args[0])
                start_node = int(args[1])
                end_node = int(args[2])
                set_flag = int(args[3])

                # Calculate length from node coordinates
                if start_node not in self.nodes or end_node not in self.nodes:
                    raise ValueError(f"Nodes {start_node} or {end_node} not found for element {elem_id}.")

                coord1 = self.nodes[start_node]
                coord2 = self.nodes[end_node]
                length = np.linalg.norm(coord2 - coord1)

                self.elements[elem_id] = {
                    'nodes': (start_node, end_node),
                    'set': set_flag,
                    'length': length
                }

            elif keyword == 'ELGEN':
                base_elem_id = int(args[0])     # ID of the element to copy
                count = int(args[1])            # Number of copies to generate
                increment = int(args[2])        # Increment in element ID for each copy

                if base_elem_id not in self.elements:
                    raise ValueError(f"Base element {base_elem_id} not found for ELGEN.")

                base_element = self.elements[base_elem_id]
                start_node, end_node = base_element['nodes']
                set_flag = base_element['set']
                length = base_element['length']  # Copy the length from the base element

                for i in range(count):
                    new_elem_id = base_elem_id + i * increment
                    new_start_node = start_node + i
                    new_end_node = end_node + i

                    self.elements[new_elem_id] = {
                        'nodes': (new_start_node, new_end_node),
                        'set': set_flag,
                        'length': length
                    }


            elif keyword == 'RIGID':
                node1 = int(args[0])
                node2 = int(args[1])
                dofs = list(map(int, args[2:]))
                self.links.append({
                    'type': 'RIGID',
                    'nodes': (node1, node2),
                    'dofs': dofs
                })

            elif keyword == 'SPRING':
                node1 = int(args[0])
                node2 = int(args[1])
                dof_stiffness = {}
                remaining = args[2:]
                if len(remaining) % 2 != 0:
                    raise ValueError("SPRING link requires pairs of DOF and stiffness values.")
                for i in range(0, len(remaining), 2):
                    dof = int(remaining[i])
                    stiffness = float(remaining[i + 1])
                    dof_stiffness[dof] = stiffness
                self.links.append({
                    'type': 'SPRING',
                    'nodes': (node1, node2),
                    'dof_stiffness': dof_stiffness
                })

        # Update node index map
        self.node_index_map = {node_id: idx for idx, node_id in enumerate(sorted(self.nodes.keys()), 1)}

    def add_element_set(self, set_id, element_type, props, **kwargs):
        """Add element set with specific type"""
        element_type = element_type.upper()

        if element_type not in self.element_handlers:
            raise ValueError(f"Unknown element type: {element_type}. Available: {list(self.element_handlers.keys())}")

        self.element_sets[set_id] = {
            'type': element_type,
            'props': props,
            **kwargs
        }


    def _handle_beam_element(self, elem_id, elem, props, **kwargs):
        """Handle 3D beam element assembly"""
        L = elem['length']

        # Handle different property formats
        if len(props) == 7:
            # BEAM format: [E, A, Iy, Iz, G, J, rho]
            E, A, Iy, Iz, G, J, rho = props
        elif len(props) == 3:
            # TRUSS format used for beam - set defaults for missing properties
            E, A, rho = props
            Iy = Iz = 1e-6  # Small default bending inertia
            G = E / (2 * (1 + 0.3))  # Default shear modulus
            J = 1e-6  # Small default torsional constant
            print(f"Warning: Beam element {elem_id} using TRUSS properties, setting defaults for Iy, Iz, G, J")
        else:
            raise ValueError(f"BEAM element expects 3 or 7 properties, got {len(props)}")

        shear_deform = kwargs.get('shear_deform', False)

        # Use your existing beam stiffness matrix
        k = self.local_stiffness_matrix_3DbeamTimoshenko(E, A, Iy, Iz, G, J, rho, L, shear_deform)
        m = self._beam_mass_matrix(rho, A, Iy, Iz, J, L)

        return k, m

    def _handle_truss_element(self, elem_id, elem, props, **kwargs):
        """Handle truss/cable element assembly"""
        L = elem['length']

        # Handle different property formats
        if len(props) == 3:
            # TRUSS format: [E, A, rho]
            E, A, rho = props
            Iy = Iz = G = J = 0.0  # Set defaults for unused properties
        elif len(props) == 7:
            # BEAM format: [E, A, Iy, Iz, G, J, rho]
            E, A, Iy, Iz, G, J, rho = props
        else:
            raise ValueError(f"TRUSS element expects 3 or 7 properties, got {len(props)}")

        # Truss stiffness - only axial terms
        k = self._truss_stiffness_matrix(E, A, L)
        m = self._truss_mass_matrix(rho, A, L)

        return k, m

    def _truss_stiffness_matrix(self, E, A, L):
        """Truss element stiffness matrix - only axial DOFs"""
        k_local = np.zeros((12, 12))

        # Only axial stiffness (DOF 0 and 6)
        k_axial = E * A / L
        k_local[0, 0] = k_axial
        k_local[6, 6] = k_axial
        k_local[0, 6] = -k_axial
        k_local[6, 0] = -k_axial

        # Add tiny stiffness to prevent singularity (1e-12 of axial stiffness)
        tiny_stiffness = k_axial * 1e-12
        for i in range(12):
            if i not in [0, 6]:
                k_local[i, i] = tiny_stiffness

        return k_local

    def _truss_mass_matrix(self, rho, A, L):
        """Truss element mass matrix - only translational masses"""
        m_local = np.zeros((12, 12))

        # Total element mass
        total_mass = rho * A * L
        nodal_mass = total_mass / 2

        # Only translational masses
        for i in [0, 1, 2]:      # Node 1 translations
            m_local[i, i] = nodal_mass
        for i in [6, 7, 8]:      # Node 2 translations
            m_local[i, i] = nodal_mass

        # Tiny masses to rotational DOFs to prevent singularity
        tiny_mass = nodal_mass * 1e-12
        for i in [3, 4, 5, 9, 10, 11]:
            m_local[i, i] = tiny_mass

        return m_local

    def _beam_mass_matrix(self, rho, A, Iy, Iz, J, L):
        m_local = np.zeros((12, 12))

        # Compute lumped mass
        m_lumped = rho * A * L
        m_node = m_lumped / 2

        # Compute lumped rotational inertia
        I_rot_y = rho * Iy * L / 2
        I_rot_z = rho * Iz * L / 2
        I_torsion = rho * (Iy + Iz) * L / 2

        # Translational masses
        for i in [0, 1, 2, 6, 7, 8]:
            m_local[i, i] = m_node

        # Rotational inertias
        m_local[3, 3] = I_torsion
        m_local[9, 9] = I_torsion
        m_local[4, 4] = I_rot_z
        m_local[10, 10] = I_rot_z
        m_local[5, 5] = I_rot_y
        m_local[11, 11] = I_rot_y

        return m_local

########### ::::::::::::::::::::::::::::::::::::::::::::::::::::: NLGEOM

    def _get_initial_strain_force(self, elem_id, elem, u_full=None):
        """Calculate the force due to initial strain (prestress)"""
        set_id = elem['set']
        element_set = self.element_sets[set_id]
        props = element_set['props']
        initial_strain = element_set.get('initial_strain', 0.0)

        # Handle different property formats
        if len(props) == 3:
            E, A, rho = props
            # Set defaults for unused properties
            Iy = Iz = G = J = 0.0
        elif len(props) == 7:
            E, A, Iy, Iz, G, J, rho = props
        else:
            raise ValueError(f"Element expects 3 or 7 properties, got {len(props)}")

        L0 = elem['length']  # Original length

        # Rest of method remains the same...
        initial_force = E * A * initial_strain

        # If we have displacements, calculate current strain
        if u_full is not None:
            n1, n2 = elem['nodes']
            node_idx1 = self.node_index_map[n1]
            node_idx2 = self.node_index_map[n2]

            base_dof1 = (node_idx1-1)*self.dof
            base_dof2 = (node_idx2-1)*self.dof

            # Current positions
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]

            u1 = u_full[base_dof1:base_dof1+3]
            u2 = u_full[base_dof2:base_dof2+3]

            current_pos1 = np.array([x1, y1, z1]) + u1
            current_pos2 = np.array([x2, y2, z2]) + u2

            L_current = np.linalg.norm(current_pos2 - current_pos1)

            # Current strain (engineering strain)
            current_strain = (L_current - L0) / L0 if L0 > 0 else 0
            #print('strain:',current_strain)

            # Total force = initial prestress force + elastic force from deformation
            elastic_force = E * A * current_strain
            total_force = initial_force + elastic_force
        else:
            # Initial state - only prestress force
            total_force = initial_force

        return total_force, initial_strain

    def _get_current_element_force_and_strain(self, elem_id, elem, u_full=None):
        """Calculate current total force and strain in element"""
        set_id = elem['set']
        element_set = self.element_sets[set_id]
        props = element_set['props']
        prestress_strain = element_set.get('initial_strain', 0.0)

        # Handle different property formats
        if len(props) == 3:
            E, A, rho = props
            # Set defaults for unused properties
            Iy = Iz = G = J = 0.0
        elif len(props) == 7:
            E, A, Iy, Iz, G, J, rho = props
        else:
            raise ValueError(f"Element expects 3 or 7 properties, got {len(props)}")

        L0 = elem['length']

        prestress_force = E * A * prestress_strain

        if u_full is not None:
            n1, n2 = elem['nodes']
            node_idx1 = self.node_index_map[n1]
            node_idx2 = self.node_index_map[n2]

            base_dof1 = (node_idx1-1)*self.dof
            base_dof2 = (node_idx2-1)*self.dof

            # Current positions with displacements
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]

            u1 = u_full[base_dof1:base_dof1+3]
            u2 = u_full[base_dof2:base_dof2+3]

            current_pos1 = np.array([x1, y1, z1]) + u1
            current_pos2 = np.array([x2, y2, z2]) + u2

            L_current = np.linalg.norm(current_pos2 - current_pos1)

            # Mechanical strain from deformation
            mechanical_strain = (L_current - L0) / L0 if L0 > 0 else 0

            # Elastic force from deformation
            elastic_force = E * A * mechanical_strain

            # TOTAL values
            total_force = prestress_force + elastic_force
            total_strain = prestress_strain + mechanical_strain
            #print('Forces:', prestress_force, elastic_force)
            #print('Strains:', prestress_strain, mechanical_strain)

            return total_force, total_strain

        else:
            # Initial undeformed state
            return prestress_force, prestress_strain


    def _get_element_geometric_stiffness(self, P, L, n1, n2):
        """Debug the geometric stiffness transformation"""
        k_geo_local = np.zeros((12, 12))

        if L <= 0:
            return k_geo_local

        # Only axial terms for truss
        g = P / L
        k_geo_local[0, 0] = g
        k_geo_local[6, 6] = g
        k_geo_local[0, 6] = -g
        k_geo_local[6, 0] = -g

        print(f"DEBUG Geometric Stiffness:")
        print(f"  P={P:.1f} N, L={L:.3f} m, g=P/L={g:.1f} N/m")
        print(f"  Local K_geo non-zero entries:")
        for i in range(12):
            for j in range(12):
                if abs(k_geo_local[i, j]) > 1e-10:
                    print(f"    [{i:2d}, {j:2d}] = {k_geo_local[i, j]:.1f}")

        # Get transformation matrix
        T = self._get_transformation_matrix(n1, n2)

        # Check what local x-axis becomes in global coordinates
        # For a vertical cable, local x should be [0,0,±1]
        local_x_global = T[0:3, 0]  # First 3 rows, first column
        print(f"  Local x-axis in global coords: {local_x_global}")

        # Transform
        k_geo_global = T.T @ k_geo_local @ T

        print(f"  Global K_geo non-zero entries:")
        global_nonzero = 0
        for i in range(12):
            for j in range(12):
                if abs(k_geo_global[i, j]) > 1e-10:
                    global_nonzero += 1
                    print(f"    [{i:2d}, {j:2d}] = {k_geo_global[i, j]:.1f}")
        print(f"  Total non-zero entries: {global_nonzero}")

        return k_geo_global


    def _get_cable_local_stiffness(self, E, A, L, prestress=0):
        """Local stiffness matrix for cable element (truss + prestress)"""
        k_local = np.zeros((12, 12))

        # Axial stiffness only (truss behavior)
        k_axial = E * A / L
        k_local[0, 0] = k_local[6, 6] = k_axial
        k_local[0, 6] = k_local[6, 0] = -k_axial

        return k_local


    def assemble_geometric_stiffness_matrix(self, u_full=None):
        """Enhanced to use prestress forces correctly"""
        self.K_geo_global = np.zeros((self.size, self.size))

        for elem_id, elem in self.elements.items():
            set_id = elem['set']
            L0 = elem['length']
            n1, n2 = elem['nodes']

            # Get total force including prestress
            total_force, total_strain = self._get_current_element_force_and_strain(elem_id, elem, u_full)

            #print('assemble force strain', total_force, total_strain)

            # Store current force for output
            self.prestress_forces[elem_id] = total_force

            # Get geometric stiffness matrix
            k_geo = self._get_element_geometric_stiffness(total_force, L0, n1, n2)

            # Assembly (your existing code)
            dofs = []
            for node in [n1, n2]:
                node_index = self.node_index_map[node]
                base_dof = (node_index-1)*self.dof
                dofs.extend(range(base_dof, base_dof+self.dof))

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    self.K_geo_global[dof_i, dof_j] += k_geo[i, j]

        #np.savetxt('Kgeogloballel.txt',self.K_geo_global)

    def assemble_geometric_stiffness_matrix_old(self, u_full=None):
        """Assemble geometric stiffness matrix based on current element forces"""
        self.K_geo_global = np.zeros((self.size, self.size))

        for elem_id, elem in self.elements.items():
            set_id = elem['set']
            element_set = self.element_sets[set_id]
            props = element_set['props']
            prestress = element_set.get('prestress', 0.0)

            L0 = elem['length']  # Original length
            n1, n2 = elem['nodes']

            # Calculate current axial force = prestress + elastic force
            if u_full is not None:
                # Calculate current length and strain
                node_idx1 = self.node_index_map[n1]
                node_idx2 = self.node_index_map[n2]

                base_dof1 = (node_idx1-1)*self.dof
                base_dof2 = (node_idx2-1)*self.dof

                # Current positions
                x1, y1, z1 = self.nodes[n1]
                x2, y2, z2 = self.nodes[n2]

                u1 = u_full[base_dof1:base_dof1+3]
                u2 = u_full[base_dof2:base_dof2+3]

                current_pos1 = np.array([x1, y1, z1]) + u1
                current_pos2 = np.array([x2, y2, z2]) + u2

                L_current = np.linalg.norm(current_pos2 - current_pos1)

                # Elastic force
                E, A, Iy, Iz, G, J, rho = props
                strain = (L_current - L0) / L0
                elastic_force = E * A * strain

                total_force = prestress + elastic_force
            else:
                # Initial state - only prestress
                total_force = prestress

            # Store current force for output
            self.prestress_forces[elem_id] = total_force

            # Get geometric stiffness matrix
            k_geo = self._get_element_geometric_stiffness(total_force, L0, n1, n2)

            # Assembly
            dofs = []
            for node in [n1, n2]:
                node_index = self.node_index_map[node]
                base_dof = (node_index-1)*self.dof
                dofs.extend(range(base_dof, base_dof+self.dof))

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    self.K_geo_global[dof_i, dof_j] += k_geo[i, j]

    def solve_nonlinear_static(self, max_iter=20, tol=1e-6):
        """Newton-Raphson solver for geometric nonlinearity"""
        print("Starting nonlinear static analysis...")

        if not hasattr(self, 'free_dofs'):
            raise ValueError("Boundary conditions must be applied first")
        if not hasattr(self, 'F_global'):
            raise ValueError("Forces must be applied first")

        # Initial guess (zero displacement)
        u_current = np.zeros(self.size)
        u_free_current = u_current[self.free_dofs]

        F_reduced = self.F_global[self.free_dofs]

        for iteration in range(max_iter):
            print(f"Iteration {iteration + 1}:")

            # Assemble tangent stiffness (elastic + geometric)
            self.assemble_stiffness_matrix()
            self.assemble_geometric_stiffness_matrix(u_current)

            K_tangent = self.K_global + self.K_geo_global
            K_tangent_reduced = K_tangent[self.free_dofs][:, self.free_dofs]

            # Calculate residual forces
            internal_forces_global = K_tangent @ u_current
            internal_forces_reduced = internal_forces_global[self.free_dofs]
            residual = F_reduced - internal_forces_reduced

            residual_norm = np.linalg.norm(residual)
            print(f"  Residual norm: {residual_norm:.6e}")

            if residual_norm < tol:
                print(f"Converged after {iteration + 1} iterations")
                break

            # Solve for displacement increment
            try:
                delta_u_free = np.linalg.solve(K_tangent_reduced, residual)
            except np.linalg.LinAlgError:
                print("Matrix singular, adding stabilization")
                K_tangent_reduced += np.eye(len(K_tangent_reduced)) * 1e-9
                delta_u_free = np.linalg.solve(K_tangent_reduced, residual)

            # Update displacement
            u_free_current += delta_u_free
            u_current = np.zeros(self.size)
            u_current[self.free_dofs] = u_free_current

            # Update element lengths and forces
            self._update_element_geometry(u_current)

        else:
            print(f"Warning: Did not converge after {max_iter} iterations")

        self.u_full = u_current
        return u_current

    def _update_element_geometry(self, u_full):
        """Update element lengths based on current displacements"""
        for elem_id, elem in self.elements.items():
            n1, n2 = elem['nodes']
            node_idx1 = self.node_index_map[n1]
            node_idx2 = self.node_index_map[n2]

            base_dof1 = (node_idx1-1)*self.dof
            base_dof2 = (node_idx2-1)*self.dof

            # Current positions
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]

            u1 = u_full[base_dof1:base_dof1+3]
            u2 = u_full[base_dof2:base_dof2+3]

            current_pos1 = np.array([x1, y1, z1]) + u1
            current_pos2 = np.array([x2, y2, z2]) + u2

            # Update current length
            elem['current_length'] = np.linalg.norm(current_pos2 - current_pos1)

###########



    def _get_element_geometric_stiffness(self, P, L, n1, n2):
        """Geometric stiffness matrix for cable with transverse terms"""
        k_geo_local = np.zeros((12, 12))

        if L <= 0 or abs(P) < 1e-10:
            return k_geo_local


        g = P / L

        transverse_stiffness = g  # P/L

        # Local y-direction (transverse)
        k_geo_local[1, 1] = transverse_stiffness
        k_geo_local[7, 7] = transverse_stiffness
        k_geo_local[1, 7] = -transverse_stiffness
        k_geo_local[7, 1] = -transverse_stiffness

        # Local z-direction (transverse)
        k_geo_local[2, 2] = transverse_stiffness
        k_geo_local[8, 8] = transverse_stiffness
        k_geo_local[2, 8] = -transverse_stiffness
        k_geo_local[8, 2] = -transverse_stiffness


        # Transform to global coordinates
        T = self._get_transformation_matrix(n1, n2)
        k_geo_global = T.T @ k_geo_local @ T

        return k_geo_global
    ############


    def solve_modal_with_initial_stress(self, num_modes=10):
        """Simple version - no filtering"""
        if not hasattr(self, 'u_full'):
            raise ValueError("Must run static analysis first")

        print("Assembling geometric stiffness...")
        self.assemble_geometric_stiffness_matrix(self.u_full)

        K_total = self.K_global + self.K_geo_global
        #np.savetxt('Kglobal.txt',self.K_global)
        K_total_reduced = K_total[self.free_dofs][:, self.free_dofs]
        M_reduced = self.M_global[self.free_dofs][:, self.free_dofs]

        print(f"Solving eigenvalue problem for {K_total_reduced.shape[0]} DOFs...")

        #from scipy.linalg import eigh
        #eigenvalues, eigenvectors = eigh(K_total_reduced, M_reduced)

        #frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)

        eigenvalues, eigenvectors = eigsh(
            K_total_reduced,
            k=num_modes,
            M=M_reduced,
            sigma=0.0,  # Find modes near 0 Hz (lowest frequencies)
            which='LM',
            tol=1e-8,
            maxiter=10000
        )

        # Mass normalization
        for i in range(eigenvectors.shape[1]):
            modal_mass = eigenvectors[:, i] @ M_reduced @ eigenvectors[:, i]
            if abs(modal_mass) > 1e-20:
                eigenvectors[:, i] /= np.sqrt(abs(modal_mass))


        # Sort by ascending frequency
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Convert to frequencies (Hz) and store
        self.frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
        self.mode_shapes = np.zeros((self.K_global.shape[0], num_modes))
        self.mode_shapes[self.free_dofs, :] = eigenvectors

        return self.frequencies










    def _check_modal_matrices_with_rigid_body(self, K, M):
        """Enhanced matrix conditioning check that analyzes rigid body modes"""
        print("Checking matrix conditioning (preserving rigid body modes)...")

        try:
            if hasattr(K, 'toarray'):
                K_dense = K.toarray()
            else:
                K_dense = K

            # Compute all eigenvalues for rigid body detection
            eigvals_K = np.linalg.eigvalsh(K_dense)

            # Count rigid body modes (near-zero eigenvalues)
            rigid_body_count = np.sum(eigvals_K < 1e-6)
            min_positive_eig = np.min(eigvals_K[eigvals_K > 1e-6]) if np.any(eigvals_K > 1e-6) else 0
            max_eig_K = np.max(eigvals_K)

            print(f"Stiffness matrix analysis:")
            print(f"  - Total DOFs: {K_dense.shape[0]}")
            print(f"  - Rigid body modes detected: {rigid_body_count}")
            print(f"  - Minimum positive eigenvalue: {min_positive_eig:.2e}")
            print(f"  - Maximum eigenvalue: {max_eig_K:.2e}")

            if rigid_body_count > 0:
                print(f"  → {rigid_body_count} rigid body mode(s) present - checking boundary conditions")

        except Exception as e:
            print(f"Could not compute stiffness matrix eigenvalues: {e}")

        # Check mass matrix
        try:
            if hasattr(M, 'toarray'):
                M_dense = M.toarray()
            else:
                M_dense = M

            zero_mass_dofs = np.sum(np.diag(M_dense) == 0)
            if zero_mass_dofs > 0:
                print(f"Warning: {zero_mass_dofs} DOFs have zero mass")

        except Exception as e:
            print(f"Could not analyze mass matrix: {e}")




    def _print_rigid_body_diagnostics(self, eigenvectors, eigenvalues):
        """Provide detailed diagnostics about rigid body modes"""
        rigid_body_indices = [i for i, eigval in enumerate(eigenvalues) if eigval < 1e-6]

        if not rigid_body_indices:
            print("\n✓ No rigid body modes detected - structure appears properly constrained")
            return

        print(f"\nRIGID BODY MODE DIAGNOSTICS:")
        print(f"Found {len(rigid_body_indices)} rigid body mode(s)")

        for i, mode_idx in enumerate(rigid_body_indices):
            mode_shape = eigenvectors[:, mode_idx]
            eigval = eigenvalues[mode_idx]

            print(f"\nRigid Body Mode {mode_idx+1} (λ={eigval:.2e}):")

            # Analyze mode shape to identify type of rigid body motion
            max_dof = np.argmax(np.abs(mode_shape))
            max_value = mode_shape[max_dof]

            # Determine DOF type
            dof_type = (max_dof % 6) + 1  # 1-6: ux,uy,uz,rx,ry,rz
            dof_names = {1: 'UX', 2: 'UY', 3: 'UZ', 4: 'RX', 5: 'RY', 6: 'RZ'}

            print(f"  - Dominant DOF: {dof_names.get(dof_type, 'Unknown')} (value: {max_value:.3f})")
            print(f"  - This suggests missing constraint in {dof_names.get(dof_type, 'unknown')} direction")

            # Count significant DOFs in this mode
            significant_dofs = np.sum(np.abs(mode_shape) > 0.1)
            print(f"  - {significant_dofs} DOFs with significant participation")

########################################################################
# application of self-weight


    def set_gravity_direction(self, direction_vector, g=9.81):
        """Set gravity direction and magnitude

        Args:
            direction_vector: [x, y, z] unit vector for gravity direction
            g: Gravity acceleration (m/s²)
        """
        self.gravity_direction = np.array(direction_vector) / np.linalg.norm(direction_vector)
        self.g = g
        print(f"Gravity set: direction={self.gravity_direction}, magnitude={g} m/s²")

    def apply_nodal_force(self, node_id, force_vector):
        """Apply concentrated forces/moments at a node

        Args:
            node_id: Node ID to apply force to
            force_vector: [Fx, Fy, Fz, Mx, My, Mz] in global coordinates
        """
        if not hasattr(self, 'F_global'):
            self.size = len(self.nodes) * self.dof
            self.F_global = np.zeros(self.size)

        if node_id not in self.node_index_map:
            raise ValueError(f"Node {node_id} not found")

        node_index = self.node_index_map[node_id]
        base_dof = (node_index - 1) * self.dof

        for i in range(min(len(force_vector), self.dof)):
            self.F_global[base_dof + i] += force_vector[i]

    def apply_gravity_loads_from_mass_matrix(self):
        """Apply gravity loads using the assembled mass matrix - ADD to existing forces"""
        if not hasattr(self, 'M_global'):
            raise ValueError("Mass matrix must be assembled first")

        print("Applying gravity loads from mass matrix...")

        # Create gravity force vector: F = M * a_gravity
        gravity_acceleration = np.zeros(self.size)

        for node_id, node_index in self.node_index_map.items():
            base_dof = (node_index - 1) * self.dof
            # Only apply to translational DOFs (0,1,2)
            gravity_acceleration[base_dof:base_dof+3] = self.gravity_direction * self.g

        # F = M * a
        gravity_forces = self.M_global @ gravity_acceleration

        # ADD to total force vector (don't replace)
        if not hasattr(self, 'F_global'):
            self.F_global = gravity_forces.copy()
        else:
            self.F_global += gravity_forces  # ← ADD, don't replace!

        total_gravity_force = np.sum(np.abs(gravity_forces)) / self.g
        print(f"Total gravity load applied: {total_gravity_force:.1f} kg equivalent mass")

        return gravity_forces


    def apply_gravity_loads_lumped(self):
        """Apply gravity loads as lumped nodal forces based on element densities"""
        if not hasattr(self, 'F_gravity_lumped'):
            self.F_gravity_lumped = np.zeros(self.size)

        print("Applying lumped gravity loads from element densities...")

        total_mass = 0.0

        for elem_id, elem in self.elements.items():
            set_id = elem['set']
            element_set = self.element_sets[set_id]
            props = element_set['props']
            E, A, Iy, Iz, G, J, rho = props
            L = elem['length']

            # Element mass
            element_mass = rho * A * L
            total_mass += element_mass

            # Distribute half mass to each node
            nodal_mass = element_mass / 2
            nodal_weight = nodal_mass * self.g

            # Apply to both nodes of the element
            n1, n2 = elem['nodes']

            for node_id in [n1, n2]:
                node_index = self.node_index_map[node_id]
                base_dof = (node_index - 1) * self.dof

                # Apply force in gravity direction (only translational DOFs)
                gravity_force = self.gravity_direction * nodal_weight
                self.F_gravity_lumped[base_dof:base_dof+3] += gravity_force

        # Initialize or add to total force vector
        if not hasattr(self, 'F_global'):
            self.F_global = self.F_gravity_lumped.copy()
        else:
            self.F_global += self.F_gravity_lumped

        print(f"Total mass: {total_mass:.1f} kg")
        print(f"Total gravity load: {total_mass * self.g:.1f} N")

        return self.F_gravity_lumped

    def apply_self_weight_only(self, method='mass_matrix'):
        """Convenience method to apply only self-weight gravity loads WITHOUT resetting existing forces"""
        if not hasattr(self, 'F_global'):
            self.size = len(self.nodes) * self.dof
            self.F_global = np.zeros(self.size)

        # Store the current forces before adding gravity
        current_forces = self.F_global.copy() if hasattr(self, 'F_global') else np.zeros(self.size)

        if method == 'mass_matrix':
            if not hasattr(self, 'M_global'):
                self.assemble_mass_matrix()
            gravity_forces = self.apply_gravity_loads_from_mass_matrix()
        else:
            gravity_forces = self.apply_gravity_loads_lumped()

        # Restore the original forces and add gravity
        self.F_global = current_forces + gravity_forces

        return self.F_global


    def apply_prestress_forces(self):
        """Apply initial forces due to prestress"""
        if not hasattr(self, 'F_global'):
            self.size = len(self.nodes) * self.dof
            self.F_global = np.zeros(self.size)

        print("Applying prestress forces...")
        total_prestress = 0

        for elem_id, elem in self.elements.items():
            set_id = elem['set']
            if set_id in self.element_sets:
                element_set = self.element_sets[set_id]
                prestress = element_set.get('prestress', 0.0)

                if prestress != 0:
                    n1, n2 = elem['nodes']
                    print(f"  Element {elem_id}: applying prestress {prestress} N between nodes {n1} and {n2}")

                    # For cable elements, prestress creates axial tension
                    # This means node n1 gets force in direction FROM n1 to n2
                    # and node n2 gets equal opposite force

                    # Get the element vector to determine force direction
                    coord1 = self.nodes[n1]
                    coord2 = self.nodes[n2]
                    element_vector = coord2 - coord1
                    element_length = np.linalg.norm(element_vector)

                    if element_length > 0:
                        # Unit vector along element
                        unit_vector = element_vector / element_length

                        # Force vector in global coordinates
                        force_vector = unit_vector * prestress

                        # Apply forces to nodes
                        # Node n1: force in +direction (tension)
                        # Node n2: force in -direction (compression)
                        node_idx1 = self.node_index_map[n1]
                        node_idx2 = self.node_index_map[n2]

                        base_dof1 = (node_idx1 - 1) * self.dof
                        base_dof2 = (node_idx2 - 1) * self.dof

                        # Apply to translational DOFs (0,1,2)
                        self.F_global[base_dof1:base_dof1+3] += force_vector
                        self.F_global[base_dof2:base_dof2+3] -= force_vector

                        total_prestress += abs(prestress)

        print(self.F_global)
########################################################################

    def _get_transformation_matrix(self, n1, n2):
        """Compute proper 3D beam transformation matrix"""
        import numpy as np

        # Get node coordinates
        x1, y1, z1 = self.nodes[n1]
        x2, y2, z2 = self.nodes[n2]

        # Element vector and length
        vec = np.array([x2 - x1, y2 - y1, z2 - z1])
        L = np.linalg.norm(vec)

        # Local x-axis (axial direction along element)
        lx = vec / L

        # Reference vector for local y-axis
        global_z = np.array([0, 0, 1])
        global_x = np.array([1, 0, 0])

        # Check if lx is nearly vertical
        if np.allclose(np.abs(lx), [0, 0, 1], atol=1e-6):
            ly = np.cross(global_x, lx)
        else:
            ly = np.cross(global_z, lx)

        ly /= np.linalg.norm(ly)

        # Local z-axis completes right-handed system
        lz = np.cross(lx, ly)

        # GLOBAL-TO-LOCAL rotation matrix
        # Each ROW is a local axis in global coordinates
        R_global_to_local = np.vstack([lx, ly, lz])

        # For finite element transformation, we need LOCAL-TO-GLOBAL
        # Since rotation matrices are orthogonal: R_local_to_global = R_global_to_local^T
        R_local_to_global = R_global_to_local.T

        # Build full 12x12 transformation matrix
        T = np.zeros((12, 12))
        for i in range(4):
            T[3*i:3*i+3, 3*i:3*i+3] = R_local_to_global

        return T


    def local_stiffness_matrix_3DbeamTimoshenko(self, E, A, Iy, Iz, G, J, rho, L, shear_deform):


        # Local stiffness matrix
        k = np.zeros((12, 12))

        # Determine phi_y and phi_z based on shear deformation flag
        if shear_deform and G*A > 0:
            phi_y = 12 * E * Iz / (G * A * L**2)
            phi_z = 12 * E * Iy / (G * A * L**2)
        else:
            phi_y = 0
            phi_z = 0


        # Axial terms (unchanged)
        k[0,0] = k[6,6] = E*A/L
        k[0,6] = k[6,0] = -E*A/L

        # Torsion terms (unchanged)
        k[3,3] = k[9,9] = G*J/L
        k[3,9] = k[9,3] = -G*J/L

        # Bending z terms (now about local y, deflection in z)
        k[1,1] = k[7,7] = 12*E*Iz/(L**3*(1+phi_y))  # Swapped Iy <-> Iz
        k[1,7] = k[7,1] = -k[1,1]
        k[1,5] = k[5,1] = k[1,11] = k[11,1] = -6*E*Iz/(L**2*(1+phi_y))  # Sign flipped due to axis swap
        k[5,7] = k[7,5] = k[7,11] = k[11,7] = -k[1,5]
        k[5,5] = k[11,11] = (4+phi_y)*E*Iz/(L*(1+phi_y))  # Swapped Iy <-> Iz
        k[5,11] = k[11,5] = (2-phi_y)*E*Iz/(L*(1+phi_y))  # Swapped Iy <-> Iz

        # Bending y terms (now about local z, deflection in y)
        k[2,2] = k[8,8] = 12*E*Iy/(L**3*(1+phi_z))  # Swapped Iz <-> Iy
        k[2,8] = k[8,2] = -k[2,2]
        k[2,4] = k[4,2] = k[2,10] = k[10,2] = 6*E*Iy/(L**2*(1+phi_z))  # Sign flipped due to axis swap
        k[4,8] = k[8,4] = k[8,10] = k[10,8] = -k[2,4]
        k[4,4] = k[10,10] = (4+phi_z)*E*Iy/(L*(1+phi_z))  # Swapped Iz <-> Iy
        k[4,10] = k[10,4] = (2-phi_z)*E*Iy/(L*(1+phi_z))  # Swapped Iz <-> Iy

        return k

    def assemble_stiffness_matrix(self):
        """Enhanced assembly that handles different element types"""
        self.n_nodes = len(self.nodes)
        self.size = self.n_nodes * self.dof
        self.K_global = np.zeros((self.size, self.size))

        #print("Assembling stiffness matrix...")
        element_counts = {}

        for elem_id, elem in self.elements.items():
            set_id = elem['set']
            element_set = self.element_sets[set_id]
            element_type = element_set['type']
            props = element_set['props']

            # Count element types
            element_counts[element_type] = element_counts.get(element_type, 0) + 1

            # Get the appropriate handler
            handler = self.element_handlers[element_type]

            # Extract only the kwargs needed (exclude 'type' and 'props')
            kwargs = {key: value for key, value in element_set.items()
                     if key not in ['type', 'props']}

            k_local, _ = handler(elem_id, elem, props, **kwargs)  # ← FIXED LINE

            # Transformation and assembly
            n1, n2 = elem['nodes']
            T = self._get_transformation_matrix(n1, n2)
            k_global = T.T @ k_local @ T

            # Assembly
            dofs = []
            for node in [n1, n2]:
                node_index = self.node_index_map[node]
                base_dof = (node_index-1)*self.dof
                dofs.extend(range(base_dof, base_dof+self.dof))

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    self.K_global[dof_i, dof_j] += k_global[i, j]


        # Process links to add constraints/springs
        for link in self.links:
            node1, node2 = link['nodes']
            node_idx1 = self.node_index_map[node1]
            node_idx2 = self.node_index_map[node2]

            if link['type'] == 'RIGID':
                # Add high stiffness springs for rigid constraints
                for dof in link['dofs']:
                    dof_idx1 = (node_idx1 - 1) * self.dof + (dof - 1)
                    dof_idx2 = (node_idx2 - 1) * self.dof + (dof - 1)
                    K = 1e12  # High penalty stiffness
                    #print('RIGID',node1, node2,node_idx1,node_idx2,dof_idx1,dof_idx2)
                    # Update global stiffness matrix
                    self.K_global[dof_idx1, dof_idx1] += K
                    self.K_global[dof_idx2, dof_idx2] += K
                    self.K_global[dof_idx1, dof_idx2] -= K
                    self.K_global[dof_idx2, dof_idx1] -= K
            elif link['type'] == 'SPRING':
                # Add springs with specified stiffness
                for dof, K in link['dof_stiffness'].items():
                    dof_idx1 = (node_idx1 - 1) * self.dof + (dof - 1)
                    dof_idx2 = (node_idx2 - 1) * self.dof + (dof - 1)
                    self.K_global[dof_idx1, dof_idx1] += K
                    self.K_global[dof_idx2, dof_idx2] += K
                    self.K_global[dof_idx1, dof_idx2] -= K
                    self.K_global[dof_idx2, dof_idx1] -= K


    def apply_bcs(self, bcs):
        """Apply only boundary conditions (no imposed displacements).
        This is used for both static and modal analysis where we need the restrained stiffness matrix.

        Args:
            bcs: List of boundary conditions [[node_id, dof], ...]
        """
        self.bcs = bcs
        self.size = len(self.nodes) * self.dof
        self.free_dofs = np.ones(self.size, bool)

        # Initialize displacement vector
        self.u = np.zeros(self.size)

        # Apply fixed BCs
        for bc in self.bcs:
            node_id, dof = bc
            node_index = self.node_index_map[node_id]
            global_dof = (node_index-1)*self.dof + dof-1
            self.free_dofs[global_dof] = False

        # Create the reduced stiffness matrix for both static and modal analysis
        self.K_reduced = self.K_global[self.free_dofs][:, self.free_dofs]

        # Keep the modal analysis version as well (might be needed for eigenvalue problems)
        self.K_reduced_modal = self.K_reduced.copy()

        # Also keep track of which DOFs are fixed (for later use)
        self.fixed_dofs = ~self.free_dofs


    def apply_imposed_displacements(self, imposed_disp):
        """Apply imposed displacements after boundary conditions."""
        if not hasattr(self, 'free_dofs'):
            raise ValueError("Must apply boundary conditions first")

        size = len(self.free_dofs)
        self.u = np.zeros(size)

        # Apply original fixed BCs (set to zero)
        self.u[~self.free_dofs] = 0

        # Track which DOFs have imposed displacements
        imposed_dofs = np.zeros(size, bool)
        for imp in imposed_disp:
            node_id, value, dof = imp
            node_index = self.node_index_map[node_id]
            global_dof = (node_index-1)*self.dof + dof-1
            self.u[global_dof] = value
            self.ulabelplot = value
            self.nodeidplot = node_id
            self.nodedofplot = dof
            imposed_dofs[global_dof] = True

        # Free DOFs are those that were originally free AND not imposed
        self.free_dofs = self.free_dofs & ~imposed_dofs

        # Modified force vector
        self.F_reduced = np.zeros(np.sum(self.free_dofs))
        if np.any(~self.free_dofs):
            self.F_reduced -= self.K_global[self.free_dofs][:, ~self.free_dofs] @ self.u[~self.free_dofs]

        self.K_reduced = self.K_global[self.free_dofs][:, self.free_dofs]

    def solve(self):
        # Solve for free DOFs
        u_free = np.linalg.solve(self.K_reduced, self.F_reduced)

        # Combine with imposed displacements
        self.u_full = np.zeros(len(self.free_dofs))
        self.u_full[self.free_dofs] = u_free
        self.u_full[~self.free_dofs] = self.u[~self.free_dofs]  # Includes imposed displacements

        return self.u_full



################ ADD application of forces and solution


    def extract_displacements(self, node_ids, filename=None):
        """Extract displacements for specified nodes and optionally save to file.

        Args:
            node_ids: List of node IDs to extract
            filename: (optional) If provided, saves to this file

        Returns:
            A list of displacement arrays in format [node_id, ux, uy, uz, urx, ury, urz]
            If filename is given, saves to file with node number in first column
        """
        if not hasattr(self, 'u_full'):
            raise ValueError("Must solve the system first")

        results = []
        for node_id in node_ids:
            if node_id not in self.node_index_map:
                raise ValueError(f"Node {node_id} not found in model")

            node_index = self.node_index_map[node_id]
            base_dof = (node_index-1)*self.dof
            disp = self.u_full[base_dof:base_dof+6]  # Get all 6 DOFs

            # Format as [node_id, ux, uy, uz, urx, ury, urz]
            result_row = [float(node_id)] + list(disp)
            results.append(result_row)

        # Save to file if requested
        if filename:
            with open(filename, 'w') as f:
                for row in results:
                    # Format with node number as first column
                    formatted = [f"{int(row[0])}"] + [f"{val:.6e}" for val in row[1:]]
                    f.write(" ".join(formatted) + "\n")

        return results

    def apply_forces(self, forces):
        """Apply nodal forces/moments to the system.

        Args:
            forces: List of force/moment applications in format
                    [[node_id, Fx, Fy, Fz, Mx, My, Mz], ...]
                    (zero values can be omitted for brevity)
        """
        # Initialize force vector
        self.F_global = np.zeros(self.size)

        # Apply each force/moment
        for force in forces:
            node_id = int(force[0])
            node_index = self.node_index_map[node_id]
            base_dof = (node_index-1)*self.dof

            # Apply forces (translational DOFs)
            if len(force) > 1:
                self.F_global[base_dof] = force[1]  # Fx
            if len(force) > 2:
                self.F_global[base_dof+1] = force[2]  # Fy
            if len(force) > 3:
                self.F_global[base_dof+2] = force[3]  # Fz

            # Apply moments (rotational DOFs)
            if len(force) > 4:
                self.F_global[base_dof+3] = force[4]  # Mx
            if len(force) > 5:
                self.F_global[base_dof+4] = force[5]  # My
            if len(force) > 6:
                self.F_global[base_dof+5] = force[6]  # Mz


    def solve_static(self):
        """Solve the static problem with applied forces and boundary conditions.

        Returns:
            u_full: Full displacement vector (including fixed DOFs)
        """
        if not hasattr(self, 'free_dofs'):
            raise ValueError("Boundary conditions must be applied first")
        if not hasattr(self, 'F_global'):
            raise ValueError("Forces must be applied first")

        # Create reduced force vector
        F_reduced = self.F_global[self.free_dofs]

        # Subtract terms from fixed DOFs if any imposed displacements exist
        if hasattr(self, 'u') and np.any(~self.free_dofs):
            F_reduced -= self.K_global[self.free_dofs][:, ~self.free_dofs] @ self.u[~self.free_dofs]

        # Solve for free DOFs
        u_free = np.linalg.solve(self.K_reduced, F_reduced)

        # Combine with fixed DOFs
        self.u_full = np.zeros(self.size)
        self.u_full[self.free_dofs] = u_free
        if hasattr(self, 'u'):
            self.u_full[~self.free_dofs] = self.u[~self.free_dofs]  # Includes any imposed displacements

        return self.u_full

    def extract_internal_forces(self, elem_ids, filename='forces.txt'):
        """Extract internal forces for specified elements using consistent local axes.

        Args:
            elem_ids: List of element IDs to extract
            filename: Output file name (default: 'forces.txt')

        Returns:
            A list of force arrays in format [elem_id, N, Vy, Vz, T, My, Mz]
            (Vy = shear in local y, Vz = shear in local z)
        """
        if not hasattr(self, 'u_full'):
            raise ValueError("Must solve the system first")
        if not hasattr(self, 'K_global'):
            raise ValueError("Must assemble stiffness matrix first")

        results = []
        for elem_id in elem_ids:
            if elem_id not in self.elements:
                raise ValueError(f"Element {elem_id} not found in model")

            elem = self.elements[elem_id]
            n1, n2 = elem['nodes']

            # Get DOF indices
            dofs = []
            for node in [n1, n2]:
                base_dof = (self.node_index_map[node]-1)*self.dof
                dofs.extend(range(base_dof, base_dof+self.dof))

            # Get transformation matrix
            T = self._get_transformation_matrix(n1, n2)

            # Get element displacements in global coordinates
            u_global = self.u_full[dofs]

            # Transform to local coordinates
            u_local = T @ u_global


            # Recompute local stiffness matrix (consistent with assembly)
            set_id = elem['set']
            props = self.element_sets[set_id]['props']
            L = elem['length']
            E, A, Iy, Iz, G, J, rho = props
            shear_deform = self.element_sets[set_id].get('shear_deform', False)

            # Local stiffness matrix (consistent with assembly)

            k = self.local_stiffness_matrix_3DbeamTimoshenko(E, A, Iy, Iz, G, J, rho, L, shear_deform)
            # Calculate local forces
            f_local = k @ u_local

            # Extract forces averaged from the twon end nodes (sign convention: CHECK)
            N = (f_local[0] - f_local[6]) * 0.5    # Axial force (tension positive)
            Vy = (f_local[1] - f_local[7]) * 0.5  # Shear in local y direction (positive up)
            Vz = (f_local[2] - f_local[8]) * 0.5    # Shear in local z direction (positive according to right-hand rule)
            T = (f_local[3] - f_local[9]) * 0.5     # Torsional moment
            My = (f_local[4] - f_local[10]) * 0.5  # Bending moment about local y (positive when top in compression)
            Mz = (f_local[5] - f_local[11]) * 0.5   # Bending moment about local z

            results.append([elem_id, N, Vy, Vz, T, My, Mz])

        # Save to file
        with open(filename, 'w') as f:
            for row in results:
                formatted = [f"{int(row[0])}"] + [f"{val:.6e}" for val in row[1:]]
                f.write(" ".join(formatted) + "\n")

        return results


    def extract_internal_forces_from_files(self, elem_ids, results_dir=r'.\results\time_histories', filename='forces_from_files.txt'):
        """Extract internal forces by reading displacements from result files.

        Args:
            elem_ids: List of element IDs to process
            results_dir: Directory containing displacement files (default: 'results')
            filename: Output file for forces (default: 'forces_from_files.txt')

        Returns:
            List of [elem_id, N, Vy, Vz, T, My, Mz] for each element
        """
        if not hasattr(self, 'K_global'):
            raise ValueError("Stiffness matrix must be assembled first")

        # Create displacement file naming pattern
        disp_pattern = os.path.join(results_dir, 'displacement_node_{}_MDyn.txt')

        results = []
        for elem_id in elem_ids:
            if elem_id not in self.elements:
                raise ValueError(f"Element {elem_id} not found in model")

            elem = self.elements[elem_id]
            n1, n2 = elem['nodes']

            # Read displacements for both nodes
            u_global = np.zeros(12)
            for i, node in enumerate([n1, n2]):
                disp_file = disp_pattern.format(node)
                try:
                    with open(disp_file, 'r') as f:
                        data = list(map(float, f.readline().split()[1:]))  # Skip time
                        u_global[i*6:(i+1)*6] = data[:6]  # Store only first 6 values
                except FileNotFoundError:
                    raise ValueError(f"Displacement file not found for node {node}")
                except Exception as e:
                    raise ValueError(f"Error reading {disp_file}: {str(e)}")

            # Get transformation matrix
            T = self._get_transformation_matrix(n1, n2)

            # Transform to local coordinates
            u_local = T @ u_global

            # Get element properties
            set_id = elem['set']
            props = self.element_sets[set_id]['props']
            L = elem['length']
            E, A, Iy, Iz, G, J, rho = props
            shear_deform = self.element_sets[set_id].get('shear_deform', False)

            # Compute local stiffness matrix
            k = self.local_stiffness_matrix_3DbeamTimoshenko(E, A, Iy, Iz, G, J, rho, L, shear_deform)

            # Calculate local forces
            f_local = k @ u_local

            # Extract forces (averaged from both ends)
            N = (f_local[0] - f_local[6]) * 0.5
            Vy = (f_local[1] - f_local[7]) * 0.5
            Vz = (f_local[2] - f_local[8]) * 0.5
            T = (f_local[3] - f_local[9]) * 0.5
            My = (f_local[4] - f_local[10]) * 0.5
            Mz = (f_local[5] - f_local[11]) * 0.5

            results.append([elem_id, N, Vy, Vz, T, My, Mz])

        # Save results
        with open(filename, 'w') as f:
            f.write("# elem_id N Vy Vz T My Mz\n")
            for row in results:
                f.write(" ".join(f"{val:.6e}" for val in row) + "\n")

        return results



    def extract_internal_forces_from_MDynfiles(self, elem_ids, results_dir=r'.\results\time_histories'):
        """Extract internal forces by reading displacements from result files.

        Args:
            elem_ids: List of element IDs to process
            results_dir: Directory containing displacement files (default: 'results')
            filename: Output file for forces (default: 'forces_from_files.txt')

        Returns:
            List of [elem_id, N, Vy, Vz, T, My, Mz] for each element
        """
        #if not hasattr(self, 'K_global'):
        #    raise ValueError("Stiffness matrix must be assembled first")

        # Create displacement file naming pattern
        disp_pattern = os.path.join(results_dir, 'displacement_node_{}_MDyn.txt')

        # Create forces file naming pattern
        forces_pattern = os.path.join(results_dir, 'forces_element_{}_MDyn.txt')

        ### Read time
        elem_id = elem_ids[0]
        elem = self.elements[elem_id]
        n1, n2 = elem['nodes']
        disp_file = disp_pattern.format(n1)
        a = np.loadtxt(disp_file)
        t = a[:,0]

        # Read all elements
        for elem_id in elem_ids:
            forces_file = forces_pattern.format(elem_id)
            results = []
            if elem_id not in self.elements:
                raise ValueError(f"Element {elem_id} not found in model")

            elem = self.elements[elem_id]
            n1, n2 = elem['nodes']

            # Read displacements for both nodes
            u_global = np.zeros((len(t),12))
            disp_file = disp_pattern.format(n1)
            a = np.loadtxt(disp_file)
            u_global[:,0] = a[:,1]*1e-3
            u_global[:,1] = a[:,2]*1e-3
            u_global[:,2] = a[:,3]*1e-3
            u_global[:,3] = a[:,4]*1e-3
            u_global[:,4] = a[:,5]*1e-3
            u_global[:,5] = a[:,6]*1e-3
            disp_file = disp_pattern.format(n2)
            a = np.loadtxt(disp_file)
            u_global[:,6] = a[:,1]*1e-3
            u_global[:,7] = a[:,2]*1e-3
            u_global[:,8] = a[:,3]*1e-3
            u_global[:,9] = a[:,4]*1e-3
            u_global[:,10] = a[:,5]*1e-3
            u_global[:,11] = a[:,6]*1e-3

            # Get transformation matrix
            T = self._get_transformation_matrix(n1, n2)

            # Get element properties
            set_id = elem['set']
            props = self.element_sets[set_id]['props']
            L = elem['length']
            E, A, Iy, Iz, G, J, rho = props
            shear_deform = self.element_sets[set_id].get('shear_deform', False)

            # Compute local stiffness matrix
            k = self.local_stiffness_matrix_3DbeamTimoshenko(E, A, Iy, Iz, G, J, rho, L, shear_deform)

            for countertime in range(len(t)):

                u_global_t = u_global[countertime,:]

                # Transform to local coordinates
                u_local = T @ u_global_t

                # Calculate local forces
                f_local = k @ u_local

                # Extract forces (averaged from both ends)
                N = (f_local[0] - f_local[6]) * 0.5
                Vy = (f_local[1] - f_local[7]) * 0.5
                Vz = (f_local[2] - f_local[8]) * 0.5
                Torque = (f_local[3] - f_local[9]) * 0.5
                My = (f_local[4] - f_local[10]) * 0.5
                Mz = (f_local[5] - f_local[11]) * 0.5

                results.append([t[countertime], N, Vy, Vz, Torque, My, Mz])

            # Save results
            with open(forces_file, 'w') as f:
                f.write("# File with averaged internal forces at centre of element:\n")
                f.write("###########################\n")
                f.write("# Col 1: Time [s]\n")
                f.write("# Col 2: Axial load N [N]\n")
                f.write("# Col 3: Transverse Shear force Vy [N]\n")
                f.write("# Col 4: Vertical Shear force Vz [N]\n")
                f.write("# Col 5: Torsional moment Mxx [Nm]\n")
                f.write("# Col 7: Vertical bending moment Myy [Nm]\n")
                f.write("# Col 8: Transverse bending moment Mzz [Nm]\n")
                f.write("########################################################\n")

                for row in results:
                    f.write(" ".join(f"{val:.6e}" for val in row) + "\n")

                f.close()
        return results

################


    def assemble_mass_matrix(self):
        """Enhanced mass matrix assembly for different element types"""
        self.n_nodes = len(self.nodes)
        self.size = self.n_nodes * self.dof
        self.M_global = np.zeros((self.size, self.size))

        print("Assembling mass matrix...")

        for elem_id, elem in self.elements.items():
            set_id = elem['set']
            element_set = self.element_sets[set_id]
            element_type = element_set['type']
            props = element_set['props']

            # Get the appropriate handler
            handler = self.element_handlers[element_type]

            # Extract only the kwargs needed (exclude 'type' and 'props')
            kwargs = {key: value for key, value in element_set.items()
                     if key not in ['type', 'props']}

            _, m_local = handler(elem_id, elem, props, **kwargs)  # ← FIXED LINE

            # Transformation and assembly
            n1, n2 = elem['nodes']
            T = self._get_transformation_matrix(n1, n2)
            m_global = T.T @ m_local @ T

            # Assembly
            dofs = []
            for node in [n1, n2]:
                node_index = self.node_index_map[node]
                base_dof = (node_index-1)*self.dof
                dofs.extend(range(base_dof, base_dof+self.dof))

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    self.M_global[dof_i, dof_j] += m_global[i, j]


    def assemble_consistent_mass_matrix(self):
        """Assemble the consistent mass matrix for a 3D Euler-Bernoulli beam element."""
        self.n_nodes = len(self.nodes)
        self.size = self.n_nodes * self.dof
        self.M_global = np.zeros((self.size, self.size))

        for elem_id, elem in self.elements.items():
            set_id = elem['set']
            props = self.element_sets[set_id]['props']
            L = elem['length']
            E, A, Iy, Iz, G, J, rho = props

            m = rho * A * L  # Total mass of element




            coeff = m / 420.0
            M_local = np.zeros((12, 12))


            # === Add consistent torsional mass (Cook) === #
            I_torsion = rho * J * L
            M_local[3, 3] = 2/6 * I_torsion
            M_local[3, 9] = 1/6 * I_torsion
            M_local[9, 3] = 1/6 * I_torsion
            M_local[9, 9] = 2/6 * I_torsion

            # Longitudinal translation (X-axis)
            M_local[0, 0] = M_local[6, 6] = 140
            M_local[0, 6] = M_local[6, 0] = 70

            # Vertical translation + rotation coupling (Y-direction, bending about Z)
            yz = [
                (1, 1, 156), (1, 5, 22*L), (1, 7, 54), (1,11, -13*L),
                (5, 5, 4*L**2), (5, 7, 13*L), (5,11, -3*L**2),
                (7, 7, 156), (7,11, -22*L), (11,11, 4*L**2)
            ]

            for i, j, val in yz:
                M_local[i, j] += val
                if i != j:
                    M_local[j, i] += val

            # Horizontal translation + rotation coupling (Z-direction, bending about Y)
            zy = [
                (2, 2, 156), (2, 4, -22*L), (2, 8, 54), (2,10, 13*L),
                (4, 4, 4*L**2), (4, 8, -13*L), (4,10, -3*L**2),
                (8, 8, 156), (8,10, 22*L), (10,10, 4*L**2)
            ]

            for i, j, val in zy:
                M_local[i, j] += val
                if i != j:
                    M_local[j, i] += val

            # Torsional (X rotation) – simplified lumped torsional mass
            M_local[3, 3] = M_local[9, 9] = m * J / (2 * A * L)

            # Scale full matrix
            M_local *= coeff

            # === Transform to Global Coordinates === #
            n1, n2 = elem['nodes']
            T = self._get_transformation_matrix(n1, n2)
            M_global_elem = T.T @ M_local @ T

            # === Assemble into Global Matrix === #
            dofs = []
            for node in [n1, n2]:
                node_index = self.node_index_map[node]
                base_dof = (node_index - 1) * self.dof
                dofs.extend(range(base_dof, base_dof + self.dof))

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    self.M_global[dof_i, dof_j] += M_global_elem[i, j]

        # === Fix any zero diagonals (for stability) === #
        diag = np.diag(self.M_global)
        zero_dofs = np.where(diag == 0)[0]

        if len(zero_dofs) > 0:
            print(f"Applying stabilization to {len(zero_dofs)} zero-mass DOFs")
            mean_mass = np.mean(diag[diag > 0])
            stabilization = 1e-6 * mean_mass if mean_mass > 0 else 1e-6
            self.M_global[zero_dofs, zero_dofs] = stabilization

        # === Apply boundary conditions === #
        if hasattr(self, 'free_dofs'):
            self.M_reduced = self.M_global[self.free_dofs][:, self.free_dofs]


    def assemble_mass_matrix_given(self):
        """Assemble the lumped mass matrix for a 3D beam element from given mass matrix"""

        # Apply boundary conditions
        if hasattr(self, 'free_dofs'):
            self.M_reduced = self.M_global[self.free_dofs][:, self.free_dofs]

    def solve_modal(self, num_modes=1, small_value_threshold=1e-12):
        """
        Solve eigenvalue problem with mass-normalized mode shapes, handling small numerical artifacts.
        Returns:
            frequencies (Hz), mass-normalized mode shapes.
        """
        # Input validation
        if not hasattr(self, 'K_reduced_modal') or not hasattr(self, 'M_reduced'):
            raise ValueError("Assemble stiffness and mass matrices first")

        # Check matrix conditioning
        self._check_modal_matrices()

        try:
            # Solve using sparse solver (faster for large systems)
            eigenvalues, eigenvectors = eigsh(
                self.K_reduced_modal,
                k=num_modes,
                M=self.M_reduced,
                sigma=0.0,  # Find modes near 0 Hz (lowest frequencies)
                which='LM',
                tol=1e-8,   # Tight tolerance for accuracy
                maxiter=10000
            )

            # Mass-normalize eigenvectors (ΦᵀMΦ = I)
            for i in range(eigenvectors.shape[1]):
                modal_mass = eigenvectors[:, i] @ self.M_reduced @ eigenvectors[:, i]
                eigenvectors[:, i] /= np.sqrt(modal_mass)

            # Threshold small values to avoid numerical artifacts
            eigenvectors[np.abs(eigenvectors) < small_value_threshold] = 0.0

        except Exception as e:
            print(f"Eigensolver failed: {str(e)}. Trying dense solver...")
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(
                self.K_reduced_modal.toarray() if hasattr(self.K_reduced_modal, 'toarray') else self.K_reduced_modal,
                self.M_reduced.toarray() if hasattr(self.M_reduced, 'toarray') else self.M_reduced,
                subset_by_index=[0, num_modes-1]
            )
            # Mass-normalize for dense solution
            for i in range(eigenvectors.shape[1]):
                eigenvectors[:, i] /= np.sqrt(eigenvectors[:, i] @ self.M_reduced @ eigenvectors[:, i])

            # Threshold small values
            eigenvectors[np.abs(eigenvectors) < small_value_threshold] = 0.0

        # Sort by ascending frequency
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Convert to frequencies (Hz) and store
        self.frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
        self.mode_shapes = np.zeros((self.K_global.shape[0], num_modes))
        self.mode_shapes[self.free_dofs, :] = eigenvectors

        return self.frequencies

    def _check_modal_matrices(self):
        """Internal check for matrix validity."""
        if np.any(np.isnan(self.K_reduced_modal)) or np.any(np.isinf(self.K_reduced_modal)):
            raise ValueError("Stiffness matrix contains NaN/Inf")
        if np.any(np.isnan(self.M_reduced)) or np.any(np.isinf(self.M_reduced)):
            raise ValueError("Mass matrix contains NaN/Inf")

        # Ensure positive definiteness
        K_min = np.min(np.linalg.eigvalsh(self.K_reduced_modal))
        M_min = np.min(np.linalg.eigvalsh(self.M_reduced))
        if K_min <= 0:
            print(f"Warning: Stiffness matrix has non-positive eigenvalues (min: {K_min:.2e})")
        if M_min <= 0:
            print(f"Warning: Mass matrix has non-positive eigenvalues (min: {M_min:.2e})")
            # Apply stabilization
            diag = np.diag(self.M_reduced)
            self.M_reduced[np.diag_indices_from(self.M_reduced)] = np.where(diag > 0, diag, 1e-6*np.mean(diag[diag > 0]))

    def stabilize_system(self):
        """Apply numerical stabilization to ill-conditioned matrices."""
        # 1. Diagonal stabilization for mass matrix
        diag_M = np.diag(self.M_reduced)
        zero_mass_dofs = np.where(diag_M < 1e-10)[0]
        if len(zero_mass_dofs) > 0:
            mean_mass = np.mean(diag_M[diag_M > 0])
            stabilization = 1e-6 * mean_mass if mean_mass > 0 else 1e-6
            self.M_reduced[zero_mass_dofs, zero_mass_dofs] = stabilization

        # 2. Diagonal stabilization for stiffness matrix
        diag_K = np.diag(self.K_reduced_modal)
        zero_stiff_dofs = np.where(diag_K < 1e-10)[0]
        if len(zero_stiff_dofs) > 0:
            mean_stiff = np.mean(diag_K[diag_K > 0])
            stabilization = 1e-6 * mean_stiff if mean_stiff > 0 else 1e-6
            self.K_reduced_modal[zero_stiff_dofs, zero_stiff_dofs] = stabilization

        # 3. Unit scaling if matrices are poorly scaled
        max_K = np.max(np.abs(self.K_reduced_modal))
        max_M = np.max(np.abs(self.M_reduced))
        if max_K/max_M > 1e12 or max_M/max_K > 1e12:
            scale = np.sqrt(max_M/max_K)
            self.K_reduced_modal = self.K_reduced_modal * scale
            print(f"Applied system scaling factor: {scale:.2e}")

    def check_system_conditioning(self):
        """Check matrix conditioning before modal analysis."""
        # Check for NaN/inf
        if np.any(np.isnan(self.K_reduced_modal)) or np.any(np.isinf(self.K_reduced_modal)):
            raise ValueError("Stiffness matrix contains NaN/inf")
        if np.any(np.isnan(self.M_reduced)) or np.any(np.isinf(self.M_reduced)):
            raise ValueError("Mass matrix contains NaN/inf")

        # Compute condition numbers
        K_cond = np.linalg.cond(self.K_reduced_modal)
        M_cond = np.linalg.cond(self.M_reduced)
        print(f"Stiffness matrix condition number: {K_cond:.2e}")
        print(f"Mass matrix condition number: {M_cond:.2e}")

        # Check positive definiteness
        K_eigvals = np.linalg.eigvalsh(self.K_reduced_modal)
        M_eigvals = np.linalg.eigvalsh(self.M_reduced)
        print(f"Min stiffness eigenvalue: {np.min(K_eigvals):.2e}")
        print(f"Min mass eigenvalue: {np.min(M_eigvals):.2e}")

        # Check zero rows/columns
        zero_rows_K = np.where(~self.K_reduced_modal.any(axis=1))[0]
        zero_rows_M = np.where(~self.M_reduced.any(axis=1))[0]
        if zero_rows_K.any():
            print(f"Warning: Stiffness matrix has zero rows at: {zero_rows_K}")
        if zero_rows_M.any():
            print(f"Warning: Mass matrix has zero rows at: {zero_rows_M}")


    def plot_undeformed_model(self):

        if not hasattr(self, 'bcs'):
            raise ValueError("Apply boundary conditions first")

        self.x_coords = np.array([coord[0] for coord in self.nodes.values()])
        self.y_coords = np.array([coord[1] for coord in self.nodes.values()])
        self.z_coords = np.array([coord[2] for coord in self.nodes.values()])

        xdimEstimate = np.ptp(self.x_coords)
        ydimEstimate = np.ptp(self.y_coords)
        zdimEstimate = np.ptp(self.z_coords)
        maxdimEstimate = max([xdimEstimate,ydimEstimate,zdimEstimate])


        self.xdimEstimate = xdimEstimate
        self.ydimEstimate = ydimEstimate
        self.zdimEstimate = zdimEstimate
        self.dimensionPlot = maxdimEstimate

        triangle_size_x = self.xdimEstimate * 0.02  # adjust as needed
        triangle_size_y = triangle_size_x*0.9  # adjust as needed
        triangle_size_z = triangle_size_x*0.9  # adjust as needed


        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        #ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space


        # Original structure
        for elem_id, elem in self.elements.items():
            n1, n2 = elem['nodes']
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-')




        for node_id, dof in self.bcs:
            coord = self.nodes[node_id]

            # Default triangle size
            scale = 1.0

            if dof == 1:  # x-direction
                tip = coord
                base1 = coord + np.array([-triangle_size_x,  0, -triangle_size_z*0.8])
                base2 = coord + np.array([-triangle_size_x, 0, triangle_size_z*0.8])
                color = 'orange'

            elif dof == 2:  # y-direction
                tip = coord
                base1 = coord + np.array([triangle_size_x, triangle_size_y, 0]) * scale
                base2 = coord + np.array([-triangle_size_x, triangle_size_y, 0]) * scale
                color = 'orange'

            elif dof == 3:  # z-direction
                tip = coord
                base1 = coord + np.array([triangle_size_x, 0, -triangle_size_z]) * scale
                base2 = coord + np.array([-triangle_size_x, 0, -triangle_size_z]) * scale
                color = 'orange'

            elif dof == 4:  # rotation about x-axis
                scale = 0.7
                tip = coord
                base1 = coord + np.array([-triangle_size_x*2,  0, -triangle_size_z*0.8])*scale
                base2 = coord + np.array([-triangle_size_x*2, 0, triangle_size_z*0.8])*scale
                color = 'blue'

            elif dof == 5:  # rotation about y-axis
                scale = 0.7
                tip = coord
                base1 = coord + np.array([triangle_size_x, triangle_size_y*2, 0]) * scale
                base2 = coord + np.array([-triangle_size_x, triangle_size_y*2, 0]) * scale
                color = 'blue'

            elif dof == 6:  # rotation about z-axis
                scale = 0.7
                tip = coord
                base1 = coord + np.array([triangle_size_x, 0, -triangle_size_z*2]) * scale
                base2 = coord + np.array([-triangle_size_x, 0, -triangle_size_z*2]) * scale
                color = 'blue'

            # Add triangle if DOF is supported
            if 'tip' in locals():
                triangle = Poly3DCollection([[tip, base1, base2]], color=color, alpha=0.8)
                ax.add_collection3d(triangle)
                del tip  # clear for next loop


        maxdimEstimate = max([xdimEstimate,ydimEstimate,zdimEstimate])
        if xdimEstimate == 0:
            # Define a small, fixed Z range centered around the mean Z value
            x_center = np.mean(self.x_coords)
            x_range = self.dimensionPlot * 0.2  # or a fixed value like 1.0
            ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        if ydimEstimate == 0:
            # Define a small, fixed Y range centered around the mean Y value
            y_center = np.mean(self.y_coords)
            y_range = self.dimensionPlot * 0.2  # or a fixed value like 1.0
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        if zdimEstimate == 0:
            # Define a small, fixed Z range centered around the mean Z value
            z_center = np.mean(self.z_coords)
            z_range = self.dimensionPlot * 0.2  # or a fixed value like 1.0
            ax.set_zlim(z_center - z_range/2, z_center + z_range/2)



        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Undeformed structural model')
        #ax.legend()
        #plt.tight_layout()
        #ax.set_xticks(ax.get_xticks()[::10])
        #ax.set_yticks([])
        #ax.set_zticks(ax.get_zticks()[::2])

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "undeformed_model.pdf"#, bbox_inches = None)
        plt.savefig(results_dir + filename)
        plt.show()
        plt.close()

    def plot_undeformed_model_with_inertia(self, scale_factor, xplot, yplot, zplot):
        """Plot model with cross-section inertia visualization.

        Args:
            scale_factor: Scaling for ellipse sizes (default 0.1)
        """

        if not hasattr(self, 'bcs'):
            raise ValueError("Apply boundary conditions first")

        self.x_coords = np.array([coord[0] for coord in self.nodes.values()])
        self.y_coords = np.array([coord[1] for coord in self.nodes.values()])
        self.z_coords = np.array([coord[2] for coord in self.nodes.values()])

        xdimEstimate = np.ptp(self.x_coords)
        ydimEstimate = np.ptp(self.y_coords)
        zdimEstimate = np.ptp(self.z_coords)
        maxdimEstimate = np.ptp(np.concatenate([self.x_coords, self.y_coords, self.z_coords]))

        annotation_scale = maxdimEstimate * 0.02  # 2% of structure size

        triangle_size_x = annotation_scale
        triangle_size_y = annotation_scale * 0.9
        triangle_size_z = annotation_scale * 0.9

        self.xdimEstimate = xdimEstimate
        self.ydimEstimate = ydimEstimate
        self.zdimEstimate = zdimEstimate
        self.dimensionPlot = maxdimEstimate


        #triangle_size_x = self.xdimEstimate * 0.02  # adjust as needed
        #triangle_size_y = triangle_size_x*0.9  # adjust as needed
        #triangle_size_z = triangle_size_x*0.9  # adjust as needed

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Original structure
        for elem_id, elem in self.elements.items():
            n1, n2 = elem['nodes']
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', alpha=0.7)

        # Plot cross-section inertias as ellipses
        for elem_id, elem in self.elements.items():
            set_id = elem['set']
            element_set = self.element_sets[set_id]
            element_type = element_set['type']
            if element_type == '3D_BEAM':
                n1, n2 = elem['nodes']
                set_id = elem['set']
                Iy = self.element_sets[set_id]['props'][2]  # Iyy
                Iz = self.element_sets[set_id]['props'][3]  # Izz

                # Normalize and scale inertias for visualization
                max_I = max(Iy, Iz)
                a = scale_factor * np.sqrt(Iy/max_I)  # Semi-axis in z-direction
                b = scale_factor * np.sqrt(Iz/max_I)  # Semi-axis in y-direction

                # Get transformation matrix for orientation
                T = self._get_transformation_matrix(n1, n2)

                local_x = T[:3,0]  # Beam axial direction
                local_y = T[:3,1]  # Strong axis (lateral)
                local_z = T[:3,2]  # Weak axis (vertical)

                # Plot for both nodes of the element
                for node in [n1, n2]:
                    center = np.array(self.nodes[node])

                    # Create ellipse in local y-z plane
                    theta = np.linspace(0, 2*np.pi, 30)
                    y = b * np.cos(theta)
                    z = a * np.sin(theta)
                    x = np.zeros_like(theta)

                    # Transform to global coordinates
                    points = np.vstack((x, y, z)).T
                    transformed_points = []
                    for point in points:
                        # Rotate from local to global system
                        global_point = center + point[0]*local_x + point[1]*local_y + point[2]*local_z
                        transformed_points.append(global_point)

                    transformed_points = np.array(transformed_points)
                    ax.plot(transformed_points[:,0], transformed_points[:,1], transformed_points[:,2],
                            'b-', lw=1, alpha=0.6)

                    # Add principal axis indicators
                    major_axis = np.array([center, center + local_z*a*1.2])
                    minor_axis = np.array([center, center + local_y*b*1.2])
                    ax.plot(major_axis[:,0], major_axis[:,1], major_axis[:,2], 'r-', lw=1, alpha=0.8)
                    ax.plot(minor_axis[:,0], minor_axis[:,1], minor_axis[:,2], 'g-', lw=1, alpha=0.8)


        for node_id, dof in self.bcs:
            coord = self.nodes[node_id]

            # Default triangle size
            scale = 1.0

            if dof == 1:  # x-direction
                tip = coord
                base1 = coord + np.array([-triangle_size_x,  0, -triangle_size_z*0.8])
                base2 = coord + np.array([-triangle_size_x, 0, triangle_size_z*0.8])
                color = 'orange'

            elif dof == 2:  # y-direction
                tip = coord
                base1 = coord + np.array([triangle_size_x, triangle_size_y, 0]) * scale
                base2 = coord + np.array([-triangle_size_x, triangle_size_y, 0]) * scale
                color = 'orange'

            elif dof == 3:  # z-direction
                tip = coord
                base1 = coord + np.array([triangle_size_x, 0, -triangle_size_z]) * scale
                base2 = coord + np.array([-triangle_size_x, 0, -triangle_size_z]) * scale
                color = 'orange'

            elif dof == 4:  # rotation about x-axis
                scale = 0.7
                tip = coord
                base1 = coord + np.array([-triangle_size_x*2,  0, -triangle_size_z*0.8])*scale
                base2 = coord + np.array([-triangle_size_x*2, 0, triangle_size_z*0.8])*scale
                color = 'blue'

            elif dof == 5:  # rotation about y-axis
                scale = 0.7
                tip = coord
                base1 = coord + np.array([triangle_size_x, triangle_size_y*2, 0]) * scale
                base2 = coord + np.array([-triangle_size_x, triangle_size_y*2, 0]) * scale
                color = 'blue'

            elif dof == 6:  # rotation about z-axis
                scale = 0.7
                tip = coord
                base1 = coord + np.array([triangle_size_x, 0, -triangle_size_z*2]) * scale
                base2 = coord + np.array([-triangle_size_x, 0, -triangle_size_z*2]) * scale
                color = 'blue'

            # Add triangle if DOF is supported
            if 'tip' in locals():
                triangle = Poly3DCollection([[tip, base1, base2]], color=color, alpha=0.8)
                ax.add_collection3d(triangle)
                del tip  # clear for next loop


        # Add rigid links
        for link in self.links:
            node1, node2 = link['nodes']
            node_idx1 = self.node_index_map[node1]
            node_idx2 = self.node_index_map[node2]

            if link['type'] == 'RIGID':
                x1, y1, z1 = self.nodes[node1]
                x2, y2, z2 = self.nodes[node2]
                ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', linestyle='--', alpha=0.7)

        # Slightly larger x range, tighter y and z

        #xplott = (xplot[0],xplot[1])
        #yplott = (yplot[0],yplot[1])
        #zplott = (zplot[0],zplot[1])

        set_3d_axes_limits(ax, xplot, yplot, zplot)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        #ax.set_title('Model with Cross-Section Inertia Visualization\n(Red: Strong axis, Green: Weak axis)')

        # Save and show
        results_dir = os.path.join(self.script_dir, 'FEM/')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "model_with_inertia.pdf"))
        plt.show()
        plt.close()


    def plot_deformed_model(self, scale=1):        # For previous static analysis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Original structure
        for elem_id, elem in self.elements.items():
            n1, n2 = elem['nodes']
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', label='Original' if elem_id == 1 else "")

        # Deformed structure - using u_full which includes imposed displacements
        for elem_id, elem in self.elements.items():
            n1, n2 = elem['nodes']
            node_index1 = self.node_index_map[n1]
            node_index2 = self.node_index_map[n2]

            base_dof1 = (node_index1-1)*self.dof
            base_dof2 = (node_index2-1)*self.dof

            # Get nodal displacements from full solution
            u1 = self.u_full[base_dof1:base_dof1+3]
            u2 = self.u_full[base_dof2:base_dof2+3]

            # Original coordinates
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]

            # Deformed coordinates (including imposed displacements)
            x1d, y1d, z1d = x1 + u1[0]*scale, y1 + u1[1]*scale, z1 + u1[2]*scale
            x2d, y2d, z2d = x2 + u2[0]*scale, y2 + u2[1]*scale, z2 + u2[2]*scale

            ax.plot([x1d, x2d], [y1d, y2d], [z1d, z2d], 'r-')#, label='Deformed' if elem_id == 1 else "")

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "deformed_structure.pdf"#, bbox_inches = None)
        plt.savefig(results_dir + filename)
        #plt.show()
        plt.close()

        filename = "static_displacements.txt"
        np.savetxt(results_dir+filename, self.u_full, fmt="%.7f")


    def plot_save_deformation(self, i, scale=1):        # For previous static analysis with imposed displacements
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Original structure
        for elem_id, elem in self.elements.items():
            n1, n2 = elem['nodes']
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', label='Original' if elem_id == 1 else "")

        # Deformed structure - using u_full which includes imposed displacements
        for elem_id, elem in self.elements.items():
            n1, n2 = elem['nodes']
            node_index1 = self.node_index_map[n1]
            node_index2 = self.node_index_map[n2]

            base_dof1 = (node_index1-1)*self.dof
            base_dof2 = (node_index2-1)*self.dof

            # Get nodal displacements from full solution
            u1 = self.u_full[base_dof1:base_dof1+3]
            u2 = self.u_full[base_dof2:base_dof2+3]

            # Original coordinates
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]

            # Deformed coordinates (including imposed displacements)
            x1d, y1d, z1d = x1 + u1[0]*scale, y1 + u1[1]*scale, z1 + u1[2]*scale
            x2d, y2d, z2d = x2 + u2[0]*scale, y2 + u2[1]*scale, z2 + u2[2]*scale

            ax.plot([x1d, x2d], [y1d, y2d], [z1d, z2d], 'r-')#, label='Deformed' if elem_id == 1 else "")

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        titlelabel = 'r_'+str(i)+'; imposed displacement of '+ str(self.ulabelplot)+ ' m at node '+ str(int(self.nodeidplot))+ ' in DOF '+ str(self.nodedofplot)
        ax.set_title(titlelabel)

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "r"+str(i)+".pdf"#, bbox_inches = None)
        plt.savefig(results_dir + filename)
        #plt.show()
        plt.close()

        filename = "r"+str(i)+".txt"
        np.savetxt(results_dir+filename, self.u_full, fmt="%.7f")

    def plot_mode_shape_simp(self, mode_numbers, scale=None, xplot=(-2, 2), yplot=(-2, 2), zplot=(-5, 1), format='svg', showplot=True):
        """
        Plot multiple vibration mode shapes in 3D - UPDATED for both regular and stressed modal analysis.

        Args:
            mode_numbers: List of mode indices to plot
            scale: Scaling factor for visualization (None for automatic)
            xplot, yplot, zplot: Plot limits
            format: Image format ('svg', 'png', etc.)
            showplot: Whether to display the plot
        """
        # Check which modal results are available
        if hasattr(self, 'mode_shapes_stressed') and hasattr(self, 'frequencies_stressed'):
            # Use stressed modal results
            mode_shapes = self.mode_shapes_stressed
            frequencies = self.frequencies_stressed
            analysis_type = " (with geometric stiffness)"
            print("Using modal results with geometric stiffness")
        elif hasattr(self, 'mode_shapes') and hasattr(self, 'frequencies'):
            # Use regular modal results
            mode_shapes = self.mode_shapes
            frequencies = self.frequencies
            analysis_type = ""
            print("Using regular modal results")
        else:
            raise ValueError("Run solve_modal() or solve_modal_with_initial_stress() first")

        results_dir = os.path.join(self.script_dir, 'FEM/')
        os.makedirs(results_dir, exist_ok=True)

        xvaluesr = abs(xplot[1]-xplot[0])
        yvaluesr = abs(yplot[1]-yplot[0])
        zvaluesr = abs(zplot[1]-zplot[0])
        valuesr = max([xvaluesr,yvaluesr,zvaluesr])

        for mode_number in mode_numbers:
            if mode_number >= mode_shapes.shape[1]:
                print(f"Warning: Mode {mode_number} not available. Only {mode_shapes.shape[1]} modes computed.")
                continue

            mode_shape = mode_shapes[:, mode_number]
            ux = mode_shape[0::6]
            uy = mode_shape[1::6]
            uz = mode_shape[2::6]
            max_ux = max(abs(ux))
            max_uy = max(abs(uy))
            max_uz = max(abs(uz))
            max_disp = max([max_ux,max_uy,max_uz])

            if max_disp > 0:
                automaticscale = valuesr * 0.1 / max_disp
            else:
                automaticscale = 1.0
                print(f"Warning: Mode {mode_number} has zero displacement")

            freq = frequencies[mode_number]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot original structure
            for elem_id, elem in self.elements.items():
                n1, n2 = elem['nodes']
                x1, y1, z1 = self.nodes[n1]
                x2, y2, z2 = self.nodes[n2]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', alpha=0.7, linewidth=2,
                       label='Original' if elem_id == 1 else "")

            # Plot deformed structure
            for elem_id, elem in self.elements.items():
                n1, n2 = elem['nodes']
                node_index1 = self.node_index_map[n1]
                node_index2 = self.node_index_map[n2]

                base_dof1 = (node_index1 - 1) * self.dof
                base_dof2 = (node_index2 - 1) * self.dof

                u1 = mode_shape[base_dof1:base_dof1 + 3]
                u2 = mode_shape[base_dof2:base_dof2 + 3]

                x1, y1, z1 = self.nodes[n1]
                x2, y2, z2 = self.nodes[n2]

                if scale is not None:
                    x1d, y1d, z1d = x1 + u1[0] * scale, y1 + u1[1] * scale, z1 + u1[2] * scale
                    x2d, y2d, z2d = x2 + u2[0] * scale, y2 + u2[1] * scale, z2 + u2[2] * scale
                else:
                    x1d, y1d, z1d = x1 + u1[0] * automaticscale, y1 + u1[1] * automaticscale, z1 + u1[2] * automaticscale
                    x2d, y2d, z2d = x2 + u2[0] * automaticscale, y2 + u2[1] * automaticscale, z2 + u2[2] * automaticscale

                ax.plot([x1d, x2d], [y1d, y2d], [z1d, z2d], 'r-', linewidth=3,
                       label='Deformed' if elem_id == 1 else "")

            # Set axis limits
            ax.set_xlim(xplot)
            ax.set_ylim(yplot)
            ax.set_zlim(zplot)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Mode {mode_number + 1}{analysis_type}\nFrequency = {freq:.3f} Hz')

            # Add legend only for first element to avoid repetition
            if elem_id == 1:
                ax.legend()

            filename = f"mode_{mode_number + 1:02d}_freq_{freq:.3f}Hz." + format
            filepath = os.path.join(results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

            if showplot:
                plt.show()
            else:
                plt.close()



    def save_Phi(self,num_modes=1):
        """
        save multiple vibration modes to file ModalInformation.txt

        Args:
            num_modes: number of modes to save, from 1 to that value all included
        """
        if not hasattr(self, 'mode_shapes'):
            raise ValueError("Run solve_modal() first")

        results_dir = os.path.join(self.script_dir, 'FEM/')
        os.makedirs(results_dir, exist_ok=True)

        self.Phi = np.zeros((self.n_nodes*num_modes,6))
        self.Mn = np.zeros(num_modes)

        for mode_number in range(num_modes):
            mode_shape = self.mode_shapes[:, mode_number]
            self.Phi[mode_number*self.n_nodes:(mode_number+1)*self.n_nodes,:] = np.array(mode_shape).reshape(self.n_nodes, 6)
            self.Mn[mode_number] = mode_shape.T @ self.M_global @ mode_shape

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "ModalInformation.txt"
        np.savetxt(results_dir+filename, self.Phi, fmt="%.7f")
        filename = "modalMassCheck.txt"
        np.savetxt(results_dir+filename, self.Mn, fmt="%.7f")


    def save_NodeInformation(self):
        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "NodeInformation.txt"

        with open(results_dir+filename, 'w') as f:

            for node_id, coords in self.nodes.items():
                # Format: node_id, x, y, z
                f.write(f"{node_id}\t{coords[0]:.6f}\t{coords[1]:.6f}\t{coords[2]:.6f}\n")


    def save_BeamInformation(self):
        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "BeamInformation.txt"

        with open(results_dir+filename, 'w') as f:

            for elem_id, elem_data in self.elements.items():
                start_node = elem_data['nodes'][0]
                end_node = elem_data['nodes'][1]
                length = elem_data['length']
                set_flag = elem_data['set']

                # Format: element_id, length, start_node, end_node, set_flag
                f.write(f"{elem_id}\t{length:.3f}\t{start_node}\t{end_node}\t{set_flag}\n")


    def save_omega(self,num_modes=1,xi=0.5):
        # xi in %
        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "FrequencyInformation.txt"
        with open(results_dir+filename, "w") as f:
            for value in self.frequencies:
                f.write(f"{value}\t{xi}\n")

    def computeGammaSYNC(self,num_modes=1):
        if not hasattr(self, 'mode_shapes'):
            raise ValueError("Run solve_modal() first")

        results_dir = os.path.join(self.script_dir, 'FEM/')
        os.makedirs(results_dir, exist_ok=True)

        self.GammaSYNC = np.zeros((num_modes,self.dof))

        self.rSYNC = np.zeros((self.size,self.dof))

        for node in range(self.n_nodes):
            self.rSYNC[node*6:(node+1)*6,0] = [1,0,0,0,0,0]
            self.rSYNC[node*6:(node+1)*6,1] = [0,1,0,0,0,0]
            self.rSYNC[node*6:(node+1)*6,2] = [0,0,1,0,0,0]
            self.rSYNC[node*6:(node+1)*6,3] = [0,0,0,1,0,0]
            self.rSYNC[node*6:(node+1)*6,4] = [0,0,0,0,1,0]
            self.rSYNC[node*6:(node+1)*6,5] = [0,0,0,0,0,1]

        for mode_number in range(num_modes):
            mode_shape = self.mode_shapes[:, mode_number]
            for dof in range(self.dof):
                self.GammaSYNC[mode_number,dof] = mode_shape.T @ self.M_global @ self.rSYNC[:,dof]

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "GammaInformation.txt"   # Sync Gamma
        np.savetxt(results_dir+filename, self.GammaSYNC, fmt="%.7f")



    def computeGammaSVGM(self,num_modes=1):
        if not hasattr(self, 'mode_shapes'):
            raise ValueError("Run solve_modal() first")

        #self.NodesEarthquake = earthquakeInfoGeometry[:,0]
        #self.DirectionsEarthquake = earthquakeInfoGeometry[:,1:4]

        results_dir = os.path.join(self.script_dir, 'FEM/')
        os.makedirs(results_dir, exist_ok=True)

        # Count number of points with restrain in each direction
        self.nrestrain_directionX = 0
        self.nrestrain_directionY = 0
        self.nrestrain_directionZ = 0

        self.NxRestrain = []
        self.NyRestrain = []
        self.NzRestrain = []

        for boundaryCond in self.bcs:
            if boundaryCond[1] == 1:
                self.nrestrain_directionX += 1
                self.NxRestrain.append(boundaryCond[0])
            if boundaryCond[1] == 2:
                self.nrestrain_directionY += 1
                self.NyRestrain.append(boundaryCond[0])
            if boundaryCond[1] == 3:
                self.nrestrain_directionZ += 1
                self.NzRestrain.append(boundaryCond[0])


        self.GammaSVGM_directionX = np.zeros((num_modes,self.nrestrain_directionX))
        self.GammaSVGM_directionY = np.zeros((num_modes,self.nrestrain_directionY))
        self.GammaSVGM_directionZ = np.zeros((num_modes,self.nrestrain_directionZ))

        self.rSVGM_directionX = np.zeros((self.size,self.nrestrain_directionX))
        self.rSVGM_directionY = np.zeros((self.size,self.nrestrain_directionY))
        self.rSVGM_directionZ = np.zeros((self.size,self.nrestrain_directionZ))

        # Calculate r vectors
        cont = 0
        contX = 0
        contY = 0
        contZ = 0

        for boundaryCond in self.bcs:
            if boundaryCond[1] == 1:
                imposed_disp = [[boundaryCond[0], 1.0, 1]]  # 1-m imposed displacement
                self.apply_imposed_displacements(imposed_disp)
                displacements = self.solve()
                self.rSVGM_directionX[:,contX] = displacements
                contX += 1
                self.plot_save_deformation(cont,scale=1)

            elif boundaryCond[1] == 2:
                imposed_disp = [[boundaryCond[0], 1.0, 2]]  # 1-m imposed displacement
                self.apply_imposed_displacements(imposed_disp)
                displacements = self.solve()
                self.rSVGM_directionY[:,contY] = displacements
                contY += 1
                self.plot_save_deformation(cont,scale=1)

            elif boundaryCond[1] == 3:
                imposed_disp = [[boundaryCond[0], 1.0, 3]]  # 1-m imposed displacement
                self.apply_imposed_displacements(imposed_disp)
                displacements = self.solve()
                self.rSVGM_directionZ[:,contZ] = displacements
                contZ += 1
                self.plot_save_deformation(cont,scale=1)

            cont += 1
                #print("Static displacements:", displacements)


        # Calculate Gamma vectors
        for mode_number in range(num_modes):
            contX = 0
            contY = 0
            contZ = 0
            mode_shape = self.mode_shapes[:, mode_number]
            for boundaryCond in self.bcs:
                if boundaryCond[1] == 1:
                    self.GammaSVGM_directionX[mode_number,contX] = mode_shape.T @ self.M_global @ self.rSVGM_directionX[:,contX]
                    contX += 1
                elif boundaryCond[1] == 2:
                    self.GammaSVGM_directionY[mode_number,contY] = mode_shape.T @ self.M_global @ self.rSVGM_directionY[:,contY]
                    contY += 1
                elif boundaryCond[1] == 3:
                    self.GammaSVGM_directionZ[mode_number,contZ] = mode_shape.T @ self.M_global @ self.rSVGM_directionZ[:,contZ]
                    contZ += 1

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        filename = "rSVGMInformationX.txt"   # Sync Gamma
        np.savetxt(results_dir+filename, self.rSVGM_directionX, fmt="%.7f")
        filename = "rSVGMInformationY.txt"   # Sync Gamma
        np.savetxt(results_dir+filename, self.rSVGM_directionY, fmt="%.7f")
        filename = "rSVGMInformationZ.txt"   # Sync Gamma
        np.savetxt(results_dir+filename, self.rSVGM_directionZ, fmt="%.7f")


        filename = "GammaSVGMInformationX.txt"   # SVGM Gamma
        np.savetxt(results_dir+filename, self.GammaSVGM_directionX, fmt="%.7f")
        filename = "GammaSVGMInformationY.txt"   # SVGM Gamma
        np.savetxt(results_dir+filename, self.GammaSVGM_directionY, fmt="%.7f")
        filename = "GammaSVGMInformationZ.txt"   # Sync Gamma
        np.savetxt(results_dir+filename, self.GammaSVGM_directionZ, fmt="%.7f")


        filename = "NodeRestraintsX.txt"
        np.savetxt(results_dir+filename, self.NxRestrain, fmt="%i")
        filename = "NodeRestraintsY.txt"
        np.savetxt(results_dir+filename, self.NyRestrain, fmt="%i")
        filename = "NodeRestraintsZ.txt"
        np.savetxt(results_dir+filename, self.NzRestrain, fmt="%i")



    def computeGammaStaticSVGM(self,num_modes=1):
        if not hasattr(self, 'mode_shapes'):
            raise ValueError("Run solve_modal() first")

        # Static Gamma proposed by Camara 2025

        results_dir = os.path.join(self.script_dir, 'FEM/')
        os.makedirs(results_dir, exist_ok=True)

        self.GammaStaticSVGM_directionX = np.zeros((num_modes,self.nrestrain_directionX))
        self.GammaStaticSVGM_directionY = np.zeros((num_modes,self.nrestrain_directionY))
        self.GammaStaticSVGM_directionZ = np.zeros((num_modes,self.nrestrain_directionZ))



        # Calculate Gamma vectors
        for mode_number in range(num_modes):
            contX = 0
            contY = 0
            contZ = 0
            mode_shape = self.mode_shapes[:, mode_number]
            omega = 2*np.pi*self.frequencies[mode_number]
            for boundaryCond in self.bcs:
                if boundaryCond[1] == 1:
                    self.GammaStaticSVGM_directionX[mode_number,contX] = (mode_shape.T @ self.K_global @ self.rSVGM_directionX[:,contX])/(omega**2)
                    contX += 1
                elif boundaryCond[1] == 2:
                    #print(mode_shape.T@self.K_global)
                    #print(self.rSVGM_directionY[:,contY])
                    #asdasdsd
                    self.GammaStaticSVGM_directionY[mode_number,contY] = (mode_shape.T @ self.K_global @ self.rSVGM_directionY[:,contY])/(omega**2)
                    contY += 1
                elif boundaryCond[1] == 3:
                    self.GammaStaticSVGM_directionZ[mode_number,contZ] = (mode_shape.T @ self.K_global @ self.rSVGM_directionZ[:,contZ])/(omega**2)
                    contZ += 1

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        filename = "GammaStaticSVGMInformationX.txt"   # SVGM Gamma
        np.savetxt(results_dir+filename, self.GammaStaticSVGM_directionX, fmt="%.7f")
        filename = "GammaStaticSVGMInformationY.txt"   # SVGM Gamma
        np.savetxt(results_dir+filename, self.GammaStaticSVGM_directionY, fmt="%.7f")
        filename = "GammaStaticSVGMInformationZ.txt"   # Sync Gamma
        np.savetxt(results_dir+filename, self.GammaStaticSVGM_directionZ, fmt="%.7f")

    def save_stiffness_matrix(self):

        sparse_matrix_K = sparse.coo_matrix(self.K_global)

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        filename = "stiffness_matrix.npz"
        sparse.save_npz(results_dir+filename, sparse_matrix_K)

    def save_mass_matrix(self):

        sparse_matrix_M = sparse.coo_matrix(self.M_global)

        results_dir = os.path.join(self.script_dir, 'FEM/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        filename = "mass_matrix.npz"
        sparse.save_npz(results_dir+filename, sparse_matrix_M)


    def _set_3d_axes_equal(self, ax):
        """Ensure 3D axes have equal scale."""
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([center[0] - radius, center[0] + radius])
        ax.set_ylim3d([center[1] - radius, center[1] + radius])
        ax.set_zlim3d([center[2] - radius, center[2] + radius])


    def _set_3d_axes_equal2(self, ax):
        """Set equal aspect ratio for 3D plot"""
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:,1] - limits[:,0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


    def _set_3d_axes_equal_and_zoom(self, ax, padding=0.1):
        """
        Set 3D plot limits based on node coordinates with optional padding.

        Args:
            ax: The 3D axis object.
            padding: Fraction of axis range to pad on each side (e.g., 0.2 = 20%).
        """
        x_coords = self.x_coords
        y_coords = self.y_coords
        z_coords = self.z_coords

        # Compute min/max for each axis
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)

        # Compute ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # Determine the maximum range to make it a cube
        max_range = max(x_range, y_range, z_range)

        # Expand limits to maintain a cube and include padding
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        # Total length including padding
        total_range = max_range * (1 + 2 * padding)

        ax.set_xlim([x_center - total_range / 2, x_center + total_range / 2])
        ax.set_ylim([y_center - total_range / 2, y_center + total_range / 2])
        ax.set_zlim([z_center - total_range / 2, z_center + total_range / 2])

    def _set_3d_axes_custom_zoom(self, ax, padding=0.2):
        """
        Set 3D plot limits based on node coordinates with per-axis padding.

        Args:
            ax: The 3D axis object.
            padding: Scalar or tuple of (px, py, pz) representing padding fraction for each axis.
        """
        x_coords = self.x_coords
        y_coords = self.y_coords
        z_coords = self.z_coords

        # Allow per-axis or uniform padding
        if isinstance(padding, (int, float)):
            px = py = pz = padding
        else:
            px, py, pz = padding

        def padded_limits(data, pad):
            min_val, max_val = np.min(data), np.max(data)
            span = max_val - min_val
            if span == 0:
                # Prevent zero-range from crashing the plot
                span = 1e-6
            return min_val - pad * span, max_val + pad * span

        xlim = padded_limits(x_coords, px)
        ylim = padded_limits(y_coords, py)
        zlim = padded_limits(z_coords, pz)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)


    def _set_3d_axes_hard_limits(self, ax,
                                 x_factors=(1.2, 1.2),
                                 y_factors=(1.1, 1.1),
                                 z_factors=(1.1, 1.1)):
        """
        Set 3D plot limits with hard multipliers on min and max for each axis.

        Args:
            ax: Matplotlib 3D axis.
            x_factors, y_factors, z_factors: Tuples of (min_factor, max_factor)
                that scale the min and max coordinates of each axis.
        """
        x_min, x_max = np.min(self.x_coords), np.max(self.x_coords)
        y_min, y_max = np.min(self.y_coords), np.max(self.y_coords)
        z_min, z_max = np.min(self.z_coords), np.max(self.z_coords)

        # Handle flat ranges by artificially inflating them slightly
        def inflate_if_flat(min_val, max_val, scale=1e-3):
            if max_val - min_val == 0:
                min_val -= scale
                max_val += scale
            return min_val, max_val

        x_min, x_max = inflate_if_flat(x_min, x_max)
        y_min, y_max = inflate_if_flat(y_min, y_max)
        z_min, z_max = inflate_if_flat(z_min, z_max)

        ax.set_xlim(x_factors[0] * x_min, x_factors[1] * x_max)
        ax.set_ylim(y_factors[0] * y_min, y_factors[1] * y_max)
        ax.set_zlim(z_factors[0] * z_min, z_factors[1] * z_max)

def set_3d_axes_limits(ax, xlim, ylim, zlim):
    """
    Set exact 3D axis limits manually.
    Args:
        ax: Matplotlib 3D axis.
        xlim, ylim, zlim: Tuples like (min, max) for each axis.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
