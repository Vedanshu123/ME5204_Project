# Impedance Spectroscopy of Microstructures using FEM

This repository contains the implementation and analysis of **Impedance Spectroscopy (IS)** for polycrystalline materials using the **Finite Element Method (FEM)**. The project was developed as part of the ME5204 Finite Element Analysis course at **IIT Madras**.

## 📘 Abstract

We simulate the electrical response of polycrystalline materials using FEM to analyze impedance spectroscopy. The domain is meshed explicitly to distinguish grains and grain boundaries. The potential is computed using an implicit time integration scheme, followed by current and impedance calculations. Nyquist plots are generated to visualize impedance variation with frequency.

## 🔬 Problem Statement

Impedance Spectroscopy is a powerful technique for characterizing materials by analyzing their electrical response. This project focuses on:

- Simulating IS using FEM.
- Capturing microstructural effects (grains and grain boundaries).
- Generating Nyquist plots to study impedance behavior.

## 🧮 Governing Equations

The simulation is based on Maxwell’s continuity equation and Ohm’s law:

- **Electric Field**: $$E = -\nabla \phi$$  
- **Current Density**: $$J = \sigma E + \epsilon \frac{\partial E}{\partial t}$$  
- **Governing PDE**: $$-\nabla \cdot (\sigma \nabla \phi) + \epsilon \frac{\partial (\nabla^2 \phi)}{\partial t} = 0$$

## 🧱 FEM Formulation

- **Weak Form** derived for numerical stability.
- **Boundary Conditions**:
  - Top: Sinusoidal voltage input.
  - Bottom: Grounded (ϕ = 0).
  - Sides: Zero flux (Neumann).

## 🧩 Discretization

- **Element Type**: Linear triangular elements.
- **Integration**: Gaussian quadrature (3-point rule).
- **Meshing Tool**: Gmsh with `.geo` file input.

## ⏱️ Time Integration

- **Scheme**: Implicit time integration.
- **Discretized Equation**:  
  $$ (M + \Delta t K) \phi^{n+1} = \Delta t F + M \phi^n $$

## 📊 Numerical Experiments

### Mesh Convergence

| Mesh Factor | Nodes | Elements | Relative Error |
|-------------|-------|----------|----------------|
| 0.01        | 6827  | 13351    | 0              |
| 0.02        | 2106  | 4054     | 1.919e-3       |
| 0.05        | 502   | 925      | 5.412e-3       |
| 0.07        | 358   | 653      | 1.716e-2       |

### Temporal Convergence

- Time step: $$\Delta t = \frac{1}{20\omega}$$
- Number of steps: 50

### Impedance Calculation

- Flux: $$J = -\sigma \nabla \phi$$
- Impedance:  
  $$Z = R \cos(\Delta \phi) + j R \sin(\Delta \phi)$$

### Nyquist Plots

Nyquist plots were generated for excitation frequencies:  
$$[10^{-3}, 10^8, 4 \times 10^6, 2 \times 10^4, 2 \times 10^3] \text{ Hz}$$  
These plots exhibit semicircular behavior, validating the simulation.

## 📌 Key Observations

- Optimal mesh factor: **0.01**
- Stable time step: **∆t = 0.01**
- Nyquist plots match theoretical expectations.
- FEM accurately captures spatial and temporal behavior of IS.

## 📚 References

- Narasimhan Swaminathan, Sundararajan Natarajan, Ean Tat Ooi, *A fast and accurate numerical technique for impedance spectroscopy of microstructures*, Journal of The Electrochemical Society, 2022. DOI: 10.1149/1945-7111/ac51a2

---
