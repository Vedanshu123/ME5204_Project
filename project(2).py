import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio

# Constants
permittivity = 100
sigma = 1

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

def solve_implicit_scheme(K, M, F, nodes, dt, num_steps, V0, omega):
    num_nodes = K.shape[0]
    phi = np.zeros(num_nodes)
    A = M + K*dt
    y_0_cord = np.zeros((num_nodes,), dtype=bool)
    y_1_cord = np.zeros((num_nodes,), dtype=bool)
    print()
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

def calculate_flux_and_current(nodes, elements, phi):
    current = np.zeros(len(nodes))  # Initialize current array for each node
    for elem in elements:
        coord = nodes[elem]  # Coordinates of the element's nodes
        phi_elem = phi[elem]  # Potential at the element's nodes

        dN = np.array([[-1, -1], [1, 0], [0, 1]])  # Gradients of shape functions
        J = dN.T @ coord  # Jacobian matrix
        det_J = np.linalg.det(J)  # Determinant of the Jacobian
        inv_J = np.linalg.inv(J)  # Inverse of the Jacobian
        B = inv_J @ dN.T  # Strain-displacement matrix

        # Initialize the nodal contributions for this element
        nodal_current = np.zeros(3)

        # Loop over Gaussian quadrature points
        for gp_idx, pt in enumerate(gaussian_points):
            w = weights[gp_idx]

            # Shape functions at the Gaussian point
            N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])

            # Compute the flux at the Gaussian point
            flux_y = -sigma * (B[1, :] @ phi_elem)  # Flux in the y-direction

            # Contribution to nodal current (weighted by shape functions and area)
            nodal_current += N * flux_y * det_J * w

        # Accumulate nodal contributions into the global current vector
        for n, node in enumerate(elem):
            current[node] += nodal_current[n]

    return current

# Main Execution
geo_file = "R-25-0.geo"
mesh_size_factor = 0.01
output_mesh = f"mesh_{mesh_size_factor:.2e}.msh"
generate_mesh(geo_file, mesh_size_factor,output_mesh)
nodes, elements = read_mesh("generated_mesh.msh")

total_simulation_cycles = 5  # Number of signal cycles to simulate
frequencies = np.array([1e-3, 1e8, 4e6, 20000,2000])  # Frequencies
real_flux = []
imag_flux = []

for omega in frequencies:
    period = 2 * np.pi / omega
    dt = (1 / (20 * omega))  # Ensure at least 20 steps per period and a small upper limit
    simulation_time = total_simulation_cycles * period
    num_steps = 50  # Ensure num_steps is at least 1

    V0 = 100  # Peak voltage
    
    # Assemble matrices and solve
    K, M, F = assemble_matrices(nodes, elements)
    phi = solve_implicit_scheme(K, M, F, nodes, dt, num_steps, V0, omega)
    # Calculate current
    current = calculate_flux_and_current(nodes, elements, phi)
    
    # Perform FFT on the voltage and current
    t_steps = np.arange(num_steps) * dt  # Time steps array
    V_signal = V0 * np.sin(omega * t_steps)  # Voltage signal as a time series
    I_signal = current[:num_steps]  # Ensure current has the correct length

    V_fft = np.fft.fft(V_signal)
    I_fft = np.fft.fft(I_signal)
    frequencies_fft = np.fft.fftfreq(len(V_signal), d=dt)
    freq_idx = np.argmin(np.abs(frequencies_fft - omega))  # Closest to input frequency
    
    V_amp = np.abs(V_fft[freq_idx])
    I_amp = np.abs(I_fft[freq_idx])
    V_phase = np.angle(V_fft[freq_idx])
    I_phase = np.angle(I_fft[freq_idx])
    
    # Impedance calculation
    R = V_amp/I_amp
    phase_diff = V_phase - I_phase
    Z_real = R * np.cos(phase_diff)
    Z_imag = R * np.sin(phase_diff)
    
    real_flux.append(Z_real)
    imag_flux.append(Z_imag)


# Nyquist Plot
plt.figure(figsize=(8, 6))
plt.plot(real_flux, imag_flux, 'o-', label="Nyquist Plot")
plt.xlabel("Real Impedance (Ohm)")
plt.ylabel("Imaginary Impedance (Ohm)")
plt.title("Nyquist Plot")
plt.grid(True)
plt.legend()
plt.show()
