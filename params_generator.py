# -*- coding: utf-8 -*-
"""
This file builds the parameter structures needed for 6D Vlasov simulation solver,
    for the journal article "OpenVlasov6: Collisionless Plasma Momentum Transfer" in Physics of 
    Plasmas, by E. Comstock & A. Romero-Calvo.
It also generates the value test functions, which are used to test results against
    analytical solutions.

@author: Eric A. Comstock

v0.4.5, Eric A. Comstock, 26-Sep-2025
v0.4.4, Eric A. Comstock, 22-Sep-2025
v0.4.3.1, Eric A. Comstock, 18-Sep-2025
v0.4.3, Eric A. Comstock, 17-Sep-2025
v0.4.2, Eric A. Comstock, 16-Sep-2025
v0.4.1, Eric A. Comstock, 15-Sep-2025
v0.4, Eric A. Comstock, 2-Sep-2025
v0.3, Eric A. Comstock, 26-Aug-2025
v0.2, Eric A. Comstock, 5-Aug-2025
v0.1, Eric A. Comstock, 3-Aug-2024
v0.0, Eric A. Comstock, 1-Aug-2024
"""

import numpy as np

density_baseline = 9e10 # Density of plasma in test cases, m^-3

def generate_Earth_params(Earth_field, B_dipole, Q, E_dipole, v_x, v_y, density, v_therm, grids):
    # This function calculates the parameters needed for the full 6D Vlasov simulation
    #   of a region in Earth's orbit.
    #
    # Inputs:
    #   Earth_field     is the Earth's magnetic field in Teslas
    #   B_dipole        is the magnetic dipole in T*m^3
    #   v_x             is the velocity of the plasma horizontally in km/s
    #   v_y             is the velocity of the plasma vertically in km/s
    #   density         is the density of the plasma in ions/cm^3
    #   v_therm         is the thermal velocity of the plasma in km/s
    #   grids           is a tuple of the position and momentum grids to be used
    #                       for the rectangular elements of the FEM mesh.
    #
    # Outputs:
    #   params          is a structure containing all the data needed to run a
    #                       6D Vlasov simulation
    
    # Initialize problem by converting inputs to a new coordinate system:
    #   Units are m, ms, m_NO (30 amu), and e - the mmm unit system
    #   1 T = 3216.178 m_NO/(ms * e)
    #   1 V = 3.216178 m_NO * m^2/(ms^2 * e)
    # Because of this, numerical values for velocity and momentum are the same
    #   for ions in this system of units.
    # When the mass of the particles is different (e.g. H2, H, e-), the mass is
    #   added in the fluids object, not params. That is why m_NO is being used -
    #   it is useful for LEO, and does not prohibit use elsewhere.
    
    # Extract dimensions from the grids tuple
    grid_x, grid_p      = grids
    
    # Convert magnetic fields from SI to mmm units
    B_earth             = Earth_field * 3216.178
    
    # Spacecraft moment in mmm units
    sp_m                = 3216.178 * np.array(B_dipole)
    
    # Calculate amount that the function undershoots the central stream of plasma due to discretization
    v0_dist             = np.exp(min(abs(grid_p)) ** 2 / v_therm ** 2)
    
    # Highest density of plasma input (avg velocity)
    p_0                 = v0_dist * density * 10 ** 6 / np.pi / v_therm ** 2
    
    # Initialize params structure for input to Vlasov solver
    params              = {}
    
    # getFEM assembly language for the dirichlet condition of Earth's ionosphere
    dirichlet_conds     = '''params['p_0'] * np.exp(-0.5*((u - params['v_x'])/params['v_therm'])**2-0.5*((v - params['v_y'])/params['v_therm'])**2-0.5*((w)/params['v_therm'])**2)'''
    zeros = '''0'''
    
    # Initialize electric and magnetic fields for input into getFEM
    params['E1']        = '''-params['B_earth'] * params['v_y']'''
    params['E2']        = '''params['B_earth'] * params['v_x']'''
    params['E3']        = '0'
    params['B1']        = '''(1 / 4 / np.pi) * (3 * x * (x * params['sp_m'][0] + y * params['sp_m'][1] + z * params['sp_m'][2]) / ((x ** 2 + y ** 2 + z ** 2) ** (5 / 2)) - params['sp_m'][0] / ((x ** 2 + y ** 2 + z ** 2) ** (3 / 2)))'''
    params['B2']        = '''(1 / 4 / np.pi) * (3 * x * (x * params['sp_m'][0] + y * params['sp_m'][1] + z * params['sp_m'][2]) / ((x ** 2 + y ** 2 + z ** 2) ** (5 / 2)) - params['sp_m'][1] / ((x ** 2 + y ** 2 + z ** 2) ** (3 / 2)))'''
    params['B3']        = '''params['B_earth'] + (1 / 4 / np.pi) * (3 * x * (x * params['sp_m'][0] + y * params['sp_m'][1] + z * params['sp_m'][2]) / ((x ** 2 + y ** 2 + z ** 2) ** (5 / 2)) - params['sp_m'][2] / ((x ** 2 + y ** 2 + z ** 2) ** (3 / 2)))'''
    
    # Initialize boundary conditions for input into getFEM
    params['BCs']       = [[40, 'Dirichlet', dirichlet_conds],[41, 'Dirichlet', dirichlet_conds],
                           [46, 'Dirichlet', zeros],
                           [47, 'Dirichlet', zeros],
                           [48, 'Dirichlet', zeros],
                           [49, 'Dirichlet', zeros],
                           [50, 'Dirichlet', zeros],
                           [51, 'Dirichlet', zeros]]
    
    params['E&B fields included'] = False # This is an initial problem, not a problem with plasma fields as well. Let the simulation iterate
    
    params['p_0']       = p_0     # Particle density at center of incoming velocity dist.
    params['v_x']       = v_x     # x-velocity of incoming plasma
    params['v_y']       = v_y     # y-velocity of incoming plasma
    params['v_therm']   = v_therm # Thermal velocity of incoming plasma
    params['B_earth']   = B_earth # Magnetic field of Earth
    params['sp_m']      = sp_m    # Spacecraft magnetic moment
    return params

def params_example1():
    # This function calculates the parameters needed for the full 6D Vlasov simulation
    #   of the first example in ""Collisionless Plasma Momentum 
    #   Transfer" in Physics of Plasmas, by E. Comstock & A. Romero-Calvo, corresponding
    #   to a thermalized plasma without electromagentic fields.
    # Note that this parameter set just tests the Vlasov equation, not Vlasov-Maxwell,
    #   so just use eval3D3V with this - not the full Vlasov-Maxwell iterateEB_until_result
    #
    # Inputs: None
    #
    # Outputs:
    #   params          is a structure containing all the data needed to run a
    #                       6D Vlasov simulation
    
    params              = {}
    
    params['v_therm']   = 2.9 # Thermal velocity of incoming plasma
    
    # getFEM assembly language for the dirichlet condition of this example
    dirichlet_conds     = '''params['p_0'] * np.exp(-0.5*((u)/params['v_therm'])**2-0.5*((v)/params['v_therm'])**2-0.5*((w)/params['v_therm'])**2)'''
    
    # Initialize electric and magnetic fields for input into getFEM
    params['E1']        = '0'
    params['E2']        = '0'
    params['E3']        = '0'
    params['B1']        = '0'
    params['B2']        = '0'
    params['B3']        = '0'
    
    # Initialize boundary conditions for input into getFEM
    params['BCs']       = [[40, 'Dirichlet', dirichlet_conds],
                           [41, 'Dirichlet', dirichlet_conds],
                           [42, 'Dirichlet', dirichlet_conds],
                           [43, 'Dirichlet', dirichlet_conds],
                           [44, 'Dirichlet', dirichlet_conds],
                           [45, 'Dirichlet', dirichlet_conds]]
    
    params['E&B fields included'] = False # This is an initial problem, not a problem with plasma fields as well. Let the simulation iterate
    
    params['p_0']       = density_baseline / np.pi / params['v_therm'] ** 2 # Particle density at center of incoming velocity distribution
    params['v_x']       = 0 # x-velocity of incoming plasma
    params['v_y']       = 0 # y-velocity of incoming plasma
    return params

def params_example2():
    # This function calculates the parameters needed for the full 6D Vlasov simulation
    #   of the second example in ""Collisionless Plasma Momentum 
    #   Transfer" in Physics of Plasmas, by E. Comstock & A. Romero-Calvo, corresponding
    #   to a thermalized plasma interacting with electric fields.
    # Note that this parameter set just tests the Vlasov equation, not Vlasov-Maxwell,
    #   so just use eval3D3V with this - not the full Vlasov-Maxwell iterateEB_until_result
    #
    # Inputs: None
    #
    # Outputs:
    #   params          is a structure containing all the data needed to run a
    #                       6D Vlasov simulation
    params              = {}
    
    params['v_therm']   = 2.9 # Thermal velocity of incoming plasma
    
    # getFEM assembly language for the dirichlet condition of this example
    dirichlet_conds     = '''params['p_0'] * np.exp(-0.5*((u)/params['v_therm'])**2-0.5*((v)/params['v_therm'])**2-0.5*((w)/params['v_therm'])**2+x**2/200*4+y**2/2312*4+z**2/1058*4)'''
    
    # Initialize electric and magnetic fields for input into getFEM
    params['E1']        = '841*x/10000'
    params['E2']        = '841*y/115600'
    params['E3']        = '841*z/52900'
    params['B1']        = '0'
    params['B2']        = '0'
    params['B3']        = '0'
    
    # Initialize boundary conditions for input into getFEM
    params['BCs']       = [[40, 'Dirichlet', dirichlet_conds],
                           [41, 'Dirichlet', dirichlet_conds],
                           [42, 'Dirichlet', dirichlet_conds],
                           [43, 'Dirichlet', dirichlet_conds],
                           [44, 'Dirichlet', dirichlet_conds],
                           [45, 'Dirichlet', dirichlet_conds]]
    
    params['E&B fields included'] = False # This is an initial problem, not a problem with plasma fields as well. Let the simulation iterate
    
    params['p_0']       = density_baseline / np.pi / params['v_therm'] ** 2 # Particle density at center of incoming velocity distribution
    params['v_x']       = 0 # x-velocity of incoming plasma
    params['v_y']       = 0 # y-velocity of incoming plasma
    return params

def params_example3():
    # This function calculates the parameters needed for the full 6D Vlasov simulation
    #   of the third example in ""Collisionless Plasma Momentum 
    #   Transfer" in Physics of Plasmas, by E. Comstock & A. Romero-Calvo, corresponding
    #   to a nonthermal plasma acting gyrokinetically.
    # Note that this parameter set just tests the Vlasov equation, not Vlasov-Maxwell,
    #   so just use eval3D3V with this - not the full Vlasov-Maxwell iterateEB_until_result
    #
    # Inputs: None
    #
    # Outputs:
    #   params          is a structure containing all the data needed to run a
    #                       6D Vlasov simulation
    params              = {}
    
    params['v_therm']   = 2.9 # Thermal velocity of incoming plasma
    
    # getFEM assembly language for the dirichlet condition of this example
    dirichlet_conds     = '''params['p_0'] * np.exp( -0.5*(np.sqrt(((u - params['v_x'])/params['v_therm'])**2+((v - params['v_y'])/params['v_therm'])**2) - params['v_rad'] / params['v_therm'])**2 -0.5* ((w)/params['v_therm'])**2)'''
    
    # Initialize electric and magnetic fields for input into getFEM
    params['E1']        = '''-params['B'] * params['v_y']'''
    params['E2']        = '''params['B'] * params['v_x']'''
    params['E3']        = '0'
    params['B1']        = '0'
    params['B2']        = '0'
    params['B3']        = '''params['B']'''
    
    # Initialize boundary conditions for input into getFEM
    params['BCs']       = [[40, 'Dirichlet', dirichlet_conds],
                           [41, 'Dirichlet', dirichlet_conds],
                           [42, 'Dirichlet', dirichlet_conds],
                           [43, 'Dirichlet', dirichlet_conds],
                           [44, 'Dirichlet', dirichlet_conds],
                           [45, 'Dirichlet', dirichlet_conds]]
    
    params['E&B fields included'] = False # This is an initial problem, not a problem with plasma fields as well. Let the simulation iterate
    
    params['p_0']       = density_baseline / np.pi / params['v_therm'] ** 2 # Particle density at center of incoming velocity distribution
    params['v_x']       = -7.8 # x-velocity of incoming plasma
    params['v_y']       = 0 # y-velocity of incoming plasma
    params['v_rad']     = 4 # Radius of the toroidal distribution in velocity-space
    params['B']         = 80e-6 # Applied magnetic field
    return params

def params_example4():
    # This function calculates the parameters needed for the full 6D Vlasov simulation
    #   of the last example in ""Collisionless Plasma Momentum 
    #   Transfer" in Physics of Plasmas, by E. Comstock & A. Romero-Calvo, corresponding
    #   to a perpendicular plasma shock with a selectable Mach number - set to 2
    #   in the paper
    # Note that this parameter is supposed to test full Vlasov-Maxwell operation,
    #   so use the full Vlasov-Maxwell iterateEB_until_result
    #
    # Inputs: None
    #
    # Outputs:
    #   params          is a structure containing all the data needed to run a
    #                       6D Vlasov simulation
    params              = {}
    
    Mach                = 2# Incoming Mach number
    params['gamma']     = 1.4 # specific heat ratio
    T_diff              = (2 * params['gamma'] * Mach ** 2 - (params['gamma'] - 1)) * ((params['gamma'] - 1) * Mach ** 2 + 2) / ((params['gamma'] + 1) ** 2 * Mach ** 2)
    rho_diff            = (params['gamma'] + 1) * Mach ** 2 / ((params['gamma'] - 1) * Mach ** 2 + 2)
    
    # getFEM assembly language for the dirichlet condition of this example
    dirichlet_conds_upstream    = '''params['p_0'] * np.exp(-0.5*((u - params['v_x2'])/params['v_therm'])**2-0.5*((v)/params['v_therm'])**2-0.5*((w)/params['v_therm'])**2)'''
    dirichlet_conds_downstream  = str(rho_diff) + ' / (' + str(T_diff) + ''' ** 1.5) * params['p_0'] * np.exp(-0.5*((u - params['v_x2'])/params['v_therm2'])**2-0.5*((v)/params['v_therm2'])**2-0.5*((w)/params['v_therm2'])**2)'''
    dirichlet_conds_side        = dirichlet_conds_upstream + ' * (0.5 * x / np.abs(x) + 0.5) + (-0.5 * x / np.abs(x) + 0.5) * ' + dirichlet_conds_downstream
    
    # Initialize electric and magnetic fields for input into getFEM
    params['E1']        = '0'
    params['E2']        = '0'
    params['E3']        = '0'
    params['B1']        = '0'
    params['B2']        = '0'
    params['B3']        = '''params['B'] * (0.5 * x / np.abs(x+1e-6) + 0.5) + (-0.5 * x / np.abs(x+1e-6) + 0.5) * params['B'] * ''' + str(rho_diff)
    # B3 jumps by the same density difference after the shock, in accordance with how perpendicular shocks work.
    
    # Initialize boundary conditions for input into getFEM
    params['BCs']       = [[40, 'Dirichlet', dirichlet_conds_upstream],
                           [41, 'Dirichlet', dirichlet_conds_downstream],
                           [42, 'Dirichlet', dirichlet_conds_side],
                           [43, 'Dirichlet', dirichlet_conds_side],
                           [44, 'Dirichlet', dirichlet_conds_side],
                           [45, 'Dirichlet', dirichlet_conds_side]]
    
    params['E&B fields included'] = False # This is an initial problem, not a problem with plasma fields as well. Let the simulation iterate
    
    params['p_0']       = 0.2 * density_baseline / np.pi / 2.9 ** 2 # Particle density at center of incoming velocity dist.
    
    params['v_therm']   = 3 # Thermal velocity of incoming plasma
    params['v_x']       = -1 * np.sqrt(params['gamma']) * params['v_therm'] * Mach # x-velocity of incoming plasma - Mach 2
    params['v_x2']      = params['v_x'] / rho_diff # x-velocity of outgoing plasma
    params['v_y']       = 0 # y-velocity of incoming plasma
    params['v_therm2']  = np.sqrt(T_diff) * params['v_therm'] # Thermal velocity of outgoing plasma
    params['B']         = 80e-6 # Applied magnetic field (incoming, before the shock)
    params['Mach']      = Mach # Adding Mach number
    return params

def value_test(result_arrays, params, soln_number):    
    # Description
    #
    # Inputs: None
    #
    # Outputs:
    #   max_error   is the maximum error of the simulation, normalized to the scale
    #                   of the density function
    #   l2_error    is the RMS error of the simulation, normalized to the scale
    #                   of the density function
    f, x, y, z, u, v, w = result_arrays # Decompose the result_arrays structure to its base components
    
    # Define the exact solution from the formula used in each test case. Then define the error as the difference bteween the simuated and exact densities
    if soln_number      == 1:
        # Exact solution for a uniform thermalized plasma
        exact_soln      = params['p_0'] * np.exp(-0.5*((u)/params['v_therm'])**2-0.5*((v)/params['v_therm'])**2-0.5*((w)/params['v_therm'])**2)
    elif soln_number    == 2:
        # Exact solution for a thermal plasma in electric fields
        exact_soln      = params['p_0'] * np.exp(-0.5*((u)/params['v_therm'])**2-0.5*((v)/params['v_therm'])**2-0.5*((w)/params['v_therm'])**2+x**2/200*4+y**2/2312*4+z**2/1058*4)
    elif soln_number    == 3:
        # Exact solution for a nonthermal plasma in electromagnetic fields
        exact_soln      = params['p_0'] * np.exp( -0.5*(np.sqrt(((u - params['v_x'])/params['v_therm'])**2+((v - params['v_y'])/params['v_therm'])**2) - params['v_rad'] / params['v_therm'])**2 -0.5* ((w)/params['v_therm'])**2)
    elif soln_number    == 4:  
        # Exact solution for a shockwave
        Mach            = params['Mach']
        T_diff          = (2 * params['gamma'] * Mach ** 2 - (params['gamma'] - 1)) * ((params['gamma'] - 1) * Mach ** 2 + 2) / ((params['gamma'] + 1) ** 2 * Mach ** 2)
        rho_diff        = (params['gamma'] + 1) * Mach ** 2 / ((params['gamma'] - 1) * Mach ** 2 + 2)
        exact_soln      = (params['p_0'] * np.exp(-0.5*((u - params['v_x2'])/params['v_therm'])**2-0.5*((v)/params['v_therm'])**2-0.5*((w)/params['v_therm'])**2)
                        * (0.5 * x / np.abs(x+1e-6) + 0.5) + (-0.5 * x / np.abs(x+1e-6) + 0.5) * 
                               rho_diff / (T_diff ** 1.5) * params['p_0'] * np.exp(-0.5*((u - params['v_x2'])/params['v_therm2'])**2-0.5*((v)/params['v_therm2'])**2-0.5*((w)/params['v_therm2'])**2))
    
    df                  = f - exact_soln
    
    # Calculating errors based on simulated density vs analytical density
    max_error           = np.max(np.abs(df)) / np.max(f) # Maximum error, normalized to the scale of the density function
    l2_error            = np.linalg.norm(np.abs(df)) / np.max(f) / np.sqrt(len(f)) # RMS error, normalized to the scale of the density function
    return max_error, l2_error