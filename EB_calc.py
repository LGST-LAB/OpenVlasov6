# -*- coding: utf-8 -*-
"""
This file included the framework for getting the electromagnetic fields from a plasma
    density distribution in six-dimensional position/momentum space, which is done
    by summing contributions from all parts of the plasma fluid. While this is O(N^6)
    versus the number of cells on each side, so is the Vlasov simulation itself,
    and this typically takes only about 10% the time of the actual Vlasov simulation.

@author: Eric A. Comstock

v0.1, Eric A. Comstock, 26-Sep-2025
v0.0, Eric A. Comstock, 6-Aug-2025
"""

import numpy as np

# Electric vacuum premittivity and magnetic vacuum permeability
# Because of my unit normalizations:
#   Units are m, ms, m_NO (30 amu), and e - the mmm unit system
#   1 T = 3216.178 m_NO/(ms * e)
#   1 V = 3.216178 m_NO * m^2/(ms^2 * e)
# We get eps0 = 8.8541878188e-12 F/m or s^4 A^2 / (m^3 kg) or s^2 C^2 / (m^3 kg)
#   so ms^2 e^2 / (m^3 m_NO) = 5.15288e-19 F/m
#   ,thus, eps0 = 17182972 ms^2 e^2 / (m^3 m_NO)
# We also get mu0 = 1.25663706e-6 H/m or kg m / (s^2 A^2) or kg m / C^2
#   so m_NO m / e^2 = 1.94066049e12 H/m, thus, mu0 = 6.47531e-19 m_NO m / e^2

eps0 = 17182972
mu0 = 6.47531e-19

def EB_compute(result_arrays, q, grids, FEM_data):
    # This function calculates the electric and magnetic field from the plasma density
    #   and velocity distribution for the current fluid. If plasma has more than 1 fluid,
    #   use this function once for each fluid density distribution, and sum the results.
    #
    # Inputs:
    #   result_arrays   is the fluid density distribution on the FEM mesh, and the
    #                       values of the six coordinates (x, y, z, u, v, w) on that
    #                       mesh. x, y, z are position, u, v, w are velocity.
    #   q               is the charge of the particles being analysed in the current
    #                       fluid, in standard electron charges. Electrons should be -1.
    #   grids           is a tuple of the position and momentum grids to be used
    #                       for the rectangular elements of the FEM mesh.
    #   FEM_data        is True if the data needs to be output indexed per FEM node
    #                       and False if it needs to be output as 3D arrays
    #
    # Outputs:
    #   E1              is the electric field in the x-direction
    #   E2              is the electric field in the y-direction
    #   E3              is the electric field in the z-direction
    #   B1              is the magnetic field in the x-direction
    #   B2              is the magnetic field in the y-direction
    #   B3              is the magnetic field in the z-direction
    
    # Initialize problem by converting inputs to a new coordinate system:
    #   Units are m, ms, m_NO (30 amu), and e - the mmm unit system
    #   1 T = 3216.178 m_NO/(ms * e)
    #   1 V = 3.216178 m_NO * m^2/(ms^2 * e)
    # Because of this, numerical values for velocity and momentum are the same
    #   for ions in this system of units.
    
    # Unpack the result arrays. Density is particle density in m^-6 ms^3
    density, x, y, z, u, v, w = result_arrays
    
    N               = len(density) # Number of FEM grid points
    
    # Get unique and inverse points for reconstructing the 6D mesh from the 1D density
    #   and coordinate lists
    x_uniq, x_inv   = np.unique(x, return_inverse = True)
    y_uniq, y_inv   = np.unique(y, return_inverse = True)
    z_uniq, z_inv   = np.unique(z, return_inverse = True)
    u_uniq, u_inv   = np.unique(u, return_inverse = True)
    v_uniq, v_inv   = np.unique(v, return_inverse = True)
    w_uniq, w_inv   = np.unique(w, return_inverse = True)
    
    ## Compute 3D charge density and current
    
    # Extract dimensions from the grids tuple
    grid_x, grid_p  = grids
    
    Nx              = len(grid_x) # Number of points in the position grid
    Np              = len(grid_p) # Number of points in the momentum grid
    
    # Initializing outputs and intermediate variables to save memory operations
    charge_xyz      = np.zeros([Nx, Nx, Nx]) # Charge per cell (e)
    mps1_xyz        = np.zeros([Nx, Nx, Nx]) # Magnetic x-pole strength per cell (e * m / ms)
    mps2_xyz        = np.zeros([Nx, Nx, Nx]) # Magnetic y-pole strength per cell (e * m / ms)
    mps3_xyz        = np.zeros([Nx, Nx, Nx]) # Magnetic z-pole strength per cell (e * m / ms)
    
    E1_xyz          = np.zeros([Nx, Nx, Nx]) # x E-field in xyz space
    E2_xyz          = np.zeros([Nx, Nx, Nx]) # y E-field in xyz space
    E3_xyz          = np.zeros([Nx, Nx, Nx]) # z E-field in xyz space
    B1_xyz          = np.zeros([Nx, Nx, Nx]) # x B-field in xyz space
    B2_xyz          = np.zeros([Nx, Nx, Nx]) # y B-field in xyz space
    B3_xyz          = np.zeros([Nx, Nx, Nx]) # z B-field in xyz space
    
    # Iterate through every FEM point, numerically integrating their densities together
    for i in range(N):
        # Find charge for every 6D fEM cell, and add to the 3D point we are interested in
        xi                      = x_inv[i]
        yi                      = y_inv[i]
        zi                      = z_inv[i]
        charge_xyz[xi][yi][zi]  += (grid_p[1]-grid_p[0]) ** 3 * (grid_x[1]-grid_x[0]) ** 3 * density[i] * q
        
        # Find magnet pole strength for every 6D fEM cell, and add to the 3D point we are interested in
        ui                      = u_inv[i]
        vi                      = v_inv[i]
        wi                      = w_inv[i]
        mps1_xyz[xi][yi][zi]    += (grid_p[1]-grid_p[0]) ** 3 * (grid_x[1]-grid_x[0]) ** 3 * density[i] * u[i] * q
        mps2_xyz[xi][yi][zi]    += (grid_p[1]-grid_p[0]) ** 3 * (grid_x[1]-grid_x[0]) ** 3 * density[i] * v[i] * q
        mps3_xyz[xi][yi][zi]    += (grid_p[1]-grid_p[0]) ** 3 * (grid_x[1]-grid_x[0]) ** 3 * density[i] * w[i] * q
    
    # Calculate electric and magnetic fields from charge and magnetic dipole contributions
    for i_target in range(Nx):
        for j_target in range(Nx):
            for k_target in range(Nx):
                # Find cell position we are finding the E and B fields in and store in memory
                pos_target = [grid_x[i_target], grid_x[j_target], grid_x[k_target]]
                for i in range(Nx):
                    for j in range(Nx):
                        for k in range(Nx):
                            # Find cell position we are calculating the E and B contributions from
                            pos_cell = np.array([grid_x[i], grid_x[j], grid_x[k]])
                            
                            # Only add contributions from other cells
                            if np.linalg.norm(pos_target - pos_cell) >= 1e-3:
                                # Distances in the vector from the E/B source position to the target position
                                xd1 = pos_target[0] - grid_x[i]
                                xd2 = pos_target[1] - grid_x[j]
                                xd3 = pos_target[2] - grid_x[k]
                                
                                # Compute the 3D distance itself to prevent unneeded computation later
                                r2  = xd1 ** 2 + xd2 ** 2 + xd3 ** 2
                                
                                # Find E field from Coulomb potential
                                E1_xyz[i_target, j_target, k_target] += charge_xyz[i][j][k] * xd1 / (4 * np.pi * eps0) / (np.sqrt(r2) ** 3)
                                E2_xyz[i_target, j_target, k_target] += charge_xyz[i][j][k] * xd2 / (4 * np.pi * eps0) / (np.sqrt(r2) ** 3)
                                E3_xyz[i_target, j_target, k_target] += charge_xyz[i][j][k] * xd3 / (4 * np.pi * eps0) / (np.sqrt(r2) ** 3)
                                
                                # Find B field from Biot-Savart law
                                B1_xyz[i_target, j_target, k_target] += (mps2_xyz[i][j][k] * xd3 - mps3_xyz[i][j][k] * xd2) * mu0 / (4 * np.pi) / (np.sqrt(r2) ** 3)
                                B2_xyz[i_target, j_target, k_target] += (mps3_xyz[i][j][k] * xd1 - mps1_xyz[i][j][k] * xd3) * mu0 / (4 * np.pi) / (np.sqrt(r2) ** 3)
                                B3_xyz[i_target, j_target, k_target] += (mps1_xyz[i][j][k] * xd2 - mps2_xyz[i][j][k] * xd1) * mu0 / (4 * np.pi) / (np.sqrt(r2) ** 3)
      
    # Choose output strategy based on requested results
    if FEM_data:
        # Generating 6D FEM node vectors to output data as for re-insertion to getFEM on the next simualtion round
        E1 = np.zeros([N])
        E2 = np.zeros([N])
        E3 = np.zeros([N])
        B1 = np.zeros([N])
        B2 = np.zeros([N])
        B3 = np.zeros([N])
        
        for i in range(N):
            # Generate arrays indexed by FEM node instead of position
            xi     = x_inv[i]
            yi     = y_inv[i]
            zi     = z_inv[i]
            E1[i] = E1_xyz[xi][yi][zi]
            E2[i] = E2_xyz[xi][yi][zi]
            E3[i] = E3_xyz[xi][yi][zi]
            B1[i] = B1_xyz[xi][yi][zi]
            B2[i] = B2_xyz[xi][yi][zi]
            B3[i] = B3_xyz[xi][yi][zi]
        return E1, E2, E3, B1, B2, B3
    else:
        uniqs = (x_uniq, y_uniq, z_uniq)
        return E1_xyz, E2_xyz, E3_xyz, B1_xyz, B2_xyz, B3_xyz, uniqs