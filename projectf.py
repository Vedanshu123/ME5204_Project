import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio

# Constants
permittivity = 100
sigma = 1e-4
V0 = 100.0  # Voltage amplitude
omega = 1.0  # Frequency in Hz
num_steps = 10  # Time steps
dt = 1 / (20 * omega)  # Time step size

# Generate Mesh
def generate_mesh(geo_file, mesh_scale_factor, output_file):
    gmsh.initialize()
    gmsh.open(geo_file)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_scale_factor)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_file)
    gmsh.finalize()

def read_mesh(filename):
    mesh = meshio.read(filename)
    nodes = mesh.points[:, :2]
    elements = mesh.cells_dict["triangle"]
    regions = mesh.cell_data_dict["gmsh:geometrical"]["triangle"]
    return nodes, elements, regions

# Assemble Matrices
def assemble_matrices(nodes, elements, regions):
    num_nodes = nodes.shape[0]
    K1 = np.zeros((num_nodes, num_nodes))
    K2 = np.zeros((num_nodes, num_nodes))
    F = np.zeros(num_nodes)

    gaussian_points = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
    weights = np.array([1/6, 1/6, 1/6])

    for iel, elem in enumerate(elements):
        econ = elem
        coord = nodes[econ]  # Coordinates of the element nodes
        region_id = regions[iel]

        K_elem_1 = np.zeros((3, 3))
        K_elem_2 = np.zeros((3, 3))

        for gp_idx in range(len(gaussian_points)):
            pt = gaussian_points[gp_idx]
            w = weights[gp_idx]

            # Shape function values and gradients at the Gaussian point
            N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
            dN = np.array([[-1, -1], [1, 0], [0, 1]])

            # Compute the Jacobian matrix and its determinant
            J = dN.T @ coord
            det_J = np.linalg.det(J)
            inv_J = np.linalg.inv(J)

            # Gradient of shape functions in global coordinates
            B = inv_J @ dN.T

            # Compute element stiffness matrices
            K_elem_1 += (B.T @ B) * sigma * det_J * w
            K_elem_2 += (B.T @ B) * permittivity * det_J * w

        

        # Assemble element contributions into global matrices and force vector
        for a in range(3):
            for b in range(3):
                K1[econ[a], econ[b]] += K_elem_1[a, b]
                K2[econ[a], econ[b]] += K_elem_2[a, b]

    return K1, K2, F


# Apply Boundary Conditions
def apply_boundary_conditions(nodes, K, F, phi, step):
    t = step * dt
    top_voltage = V0 * np.sin(omega * t)

    for i, (x, y) in enumerate(nodes):
        if np.isclose(y, 0):  # Bottom boundary
            K[i, :] = 0
            K[:, i] = 0
            K[i, i] = 1
            F[i] = 0
        elif np.isclose(y, 1):  # Top boundary
            K[i, :] = 0
            K[:, i] = 0
            K[i, i] = 1
            F[i] = top_voltage

    return K, F, phi

# Implicit Scheme
def solve_implicit_scheme(K1, K2, nodes):
    num_nodes = K1.shape[0]
    phi = np.zeros(num_nodes)  # Initial potential distribution
    A = K1 + dt * K2
    A += np.eye(A.shape[0]) * 1e-6  # Regularization for numerical stability

    for step in range(num_steps):
        F = np.zeros(num_nodes)
        K_temp = A.copy()
        K_temp, F, phi = apply_boundary_conditions(nodes, K_temp, F, phi, step)
        rhs = F * dt + K1 @ phi
        phi = np.linalg.solve(K_temp, rhs)
        bottom_nodes = np.isclose(nodes[:, 1], 0)
        phi[bottom_nodes] = 0 
        t = step * dt 
        top_voltage = V0 * np.sin(omega * t)
        top_nodes = np.isclose(nodes[:, 1], 1)
        phi[top_nodes] = top_voltage

    return phi

# Mesh Convergence Study
# Mesh Convergence Study
def mesh_convergence(geo_file, mesh_factors, node_indices):
    errors = []
    prev_solution = None  # To store the solution from the previous mesh

    for factor in mesh_factors:
        # Generate and read mesh
        output_mesh = f"mesh_{factor:.2e}.msh"
        generate_mesh(geo_file, factor, output_mesh)
        nodes, elements, regions = read_mesh(output_mesh)

        # Assemble matrices
        K1, K2, F = assemble_matrices(nodes, elements, regions)

        # Solve implicit scheme
        phi = solve_implicit_scheme(K1, K2, nodes)

        # Extract solution at specified nodes
        selected_phi = phi[node_indices]

        if prev_solution is not None:
            # Compute relative error
            error = np.linalg.norm(selected_phi - prev_solution) / np.linalg.norm(prev_solution)
            errors.append(error)
        else:
            # No error for the first solution
            errors.append(0)

        # Update the previous solution
        prev_solution = selected_phi

    return errors


# Main Execution
geo_file = "R-25-0.geo"
mesh_factors = np.array([0.008,0.009,0.01,0.012,0.015,0.02])  # Example mesh size factors
node_indices = np.arange(25, 46) 
errors = mesh_convergence(geo_file, mesh_factors, node_indices)

# Plot Mesh Convergence
plt.figure(figsize=(8, 6))
plt.plot(mesh_factors, errors, marker='o', label="Mesh Convergence")
plt.xlabel("Mesh Characteristic Length Factor")
plt.ylabel("Relative Error")
plt.grid(True)
plt.legend()
plt.title("Mesh Convergence Study")
plt.show()
