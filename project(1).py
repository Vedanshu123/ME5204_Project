import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio

# Constants
permittivity = 100
sigma = 1
V0 = 100.0  # Voltage amplitude
omega = 1.0  # Frequency in Hz
num_steps = 51  # Time steps(50 in solving because n-1 steps)
dt = 1 / (20 * omega)

# Mesh Generation
def generate_mesh(geo_file, mesh_scale_factor,output_mesh):
    gmsh.initialize()
    gmsh.open(geo_file)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_scale_factor)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_mesh)
    gmsh.finalize()

def read_mesh(filename):
    mesh = meshio.read(filename)
    nodes = mesh.points[:, :2]
    elements = mesh.cells_dict["triangle"]
    return nodes, elements

# Gaussian Quadrature Points and Weights
gaussian_points = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
weights = np.array([1/6, 1/6, 1/6])

# Assemble Matrices
def assemble_matrices(nodes, elements):
    num_nodes = nodes.shape[0]
    K = np.zeros((num_nodes, num_nodes))
    M = np.zeros((num_nodes, num_nodes))
    F = np.zeros(num_nodes)

    for iel, elem in enumerate(elements):
        econ = elem
        coord = nodes[econ]
        # region_id = regions[iel]

        K_elem = np.zeros((3, 3))
        M_elem = np.zeros((3, 3))

        for gp_idx in range(len(gaussian_points)):
            pt = gaussian_points[gp_idx]
            w = weights[gp_idx]

            # Shape function gradients and Jacobian
            dN = np.array([[-1, -1], [1, 0], [0, 1]])
            J = dN.T @ coord
            det_J = np.linalg.det(J)
            inv_J = np.linalg.inv(J)

            B = inv_J @ dN.T

            K_elem += (B.T @ B) * sigma * det_J * w
            M_elem += (B.T @ B) * permittivity * det_J * w
        # print(K_elem_1, K_elem_2)
        for a in range(3):
            for b in range(3):
                K[econ[a], econ[b]] += K_elem[a, b]
                M[econ[a], econ[b]] += M_elem[a, b]

    return K, M, F

def solve_implicit_scheme(K, M, F,nodes):
    num_nodes = K.shape[0]
    phi = np.zeros(num_nodes)
    A = M + K*dt
    y_0_cord = np.zeros((num_nodes,), dtype=bool)
    y_1_cord = np.zeros((num_nodes,), dtype=bool)
    for node_num in range(nodes.shape[0]):
        if nodes[node_num][1] == 0:
            y_0_cord[node_num]= True
        if nodes[node_num][1] == 1:
            y_1_cord[node_num]= True

    for step in range(num_steps):
        F_temp = F.copy()
        K_temp = A.copy()
        t = step * dt
        top_voltage = V0 * np.sin(omega * t)
        rhs = F_temp*dt + M @ phi
        phi[y_0_cord] = 0
        phi[y_1_cord] = top_voltage
        phi[(~y_0_cord) & (~y_1_cord)] = np.linalg.inv(K_temp[(~y_0_cord) & (~y_1_cord)][:,(~y_0_cord) & (~y_1_cord)]).dot(rhs[(~y_0_cord) & (~y_1_cord)] - K_temp[(~y_0_cord) & (~y_1_cord)][:, ~((~y_0_cord) & (~y_1_cord))].dot(phi[~((~y_0_cord) & (~y_1_cord))]))

    return phi

def mesh_convergence(geo_file, mesh_factors, node_indices):
    errors = []
    prev_solution = None  # To store the solution from the previous mesh

    for factor in mesh_factors:
        # Generate and read mesh
        output_mesh = f"mesh_{factor:.2e}.msh"
        generate_mesh(geo_file, factor, output_mesh)
        nodes, elements = read_mesh(output_mesh)
        print(f"Mesh Factor: {factor:.2e}, Number of Nodes: {nodes.shape[0]}, Number of Elements: {elements.shape[0]}")

        # Assemble matrices
        K, M, F = assemble_matrices(nodes, elements)

        # Solve implicit scheme
        phi = solve_implicit_scheme(K, M, F,nodes)

        # Extract solution at specified nodes
        selected_phi = phi[node_indices]

        if prev_solution is not None:
            
            error = np.linalg.norm(selected_phi - prev_solution) / np.linalg.norm(prev_solution)
            errors.append(error)
            print(f"error:{error}")
              
        else:
            
            errors.append(0)

        # Update the previous solution
        prev_solution = selected_phi

    return errors

# Main Execution
geo_file = "R-25-0.geo"
mesh_factors = np.array([0.01,0.02,0.05,0.07])  # Example mesh size factors
node_indices = np.arange(0, 113) 
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

mesh_size_factors = [0.01]

for scale in mesh_size_factors:
    output_mesh = f"mesh{scale:.2e}.msh"
    generate_mesh(geo_file, scale,output_mesh)
    nodes, elements = read_mesh("generated_mesh.msh")
    phi = np.zeros(len(nodes))
    K, M, F = assemble_matrices(nodes, elements)
    phi = solve_implicit_scheme(K, M, F, nodes)
    print("Final Potential at nodes:", phi)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def plot_potential_contour(nodes, elements, phi):
    """
    Plot the contour of the potential distribution.
    """
    plt.figure(figsize=(8, 6))
    triangulation = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    plt.tricontourf(triangulation, phi, levels=100, cmap="viridis")
    plt.colorbar(label="Potential (V)")
    plt.title("Contour Plot of Potential Distribution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis("equal")
    plt.show()


def plot_potential_3d(nodes, phi):
    """
    Plot the 3D surface of the potential distribution.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(nodes[:, 0], nodes[:, 1], phi, cmap="viridis", edgecolor="none")
    ax.set_title("3D Surface Plot of Potential Distribution")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Potential (V)")
    plt.show()

phi = solve_implicit_scheme(K, M, F, nodes)

plot_potential_contour(nodes,elements,phi)

plot_potential_3d(nodes,phi)