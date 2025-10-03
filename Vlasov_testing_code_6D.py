# -*- coding: utf-8 -*-
"""
This file conducts the tests needed for 6D Vlasov simulation validation, allowing
    for the generation of the results of the journal article "OpenVlasov6: A 3D-3V Fully Kinetic
    Multifluid Vlasov Solver" in Physics of Plasmas, by E. Comstock & A. Romero-Calvo.

@author: Eric A. Comstock

v1.0, Eric A. Comstock, 3-Oct-2025
v0.4.5, Eric A. Comstock, 26-Sep-2025
v0.4.4, Eric A. Comstock, 22-Sep-2025
v0.4.3.1, Eric A. Comstock, 18-Sep-2025
v0.4.3, Eric A. Comstock, 17-Sep-2025
v0.4.2, Eric A. Comstock, 16-Sep-2025
v0.4.1, Eric A. Comstock, 15-Sep-2025
v0.4, Eric A. Comstock, 2-Sep-2025
v0.3, Eric A. Comstock, 26-Aug-2025
v0.2.7, Eric A. Comstock, 5-Aug-2025
v0.2.6.1, Eric A. Comstock, 29-Jul-2025
v0.2.6, Eric A. Comstock, 28-Jul-2025
v0.2.5.3, Eric A. Comstock, 25-Jul-2025
v0.2.5.2, Eric A. Comstock, 23-Jul-2025
v0.2.5.1, Eric A. Comstock, 16-Jul-2025
v0.2.5, Eric A. Comstock, 30-Jun-2025
v0.2.4, Eric A. Comstock, 16-Jun-2025
v0.2.3.1, Eric A. Comstock, 27-May-2025
v0.2.3, Eric A. Comstock, 02-May-2025
v0.2.2, Eric A. Comstock, 28-Apr-2025
v0.2.1, Eric A. Comstock, 24-Apr-2025
v0.2, Eric A. Comstock, 28-Feb-2025
v0.1.4, Eric A. Comstock, 11-Feb-2025
v0.1.3, Eric A. Comstock, 13-Dec-2024
v0.1.2, Eric A. Comstock, 08-Dec-2024
v0.1.1, Eric A. Comstock, 19-Nov-2024
v0.1, Eric A. Comstock, 14-Nov-2024
v0.0, Eric A. Comstock, 28-Oct-2024
"""

#### Import basic modules ####

import numpy as np              # Used for vector algebra and for getFEM
import getfem as gf             # Main FEM assembly package used in this code
import time                     # Used for getting time for logging and file names
import scipy                    # Used for matrix operations - most customizable than getFEM
import scipy.interpolate        # Used for interpolation of EM fields for more accurate FEA
import scipy.sparse             # In some OS environments, importing every specific subpackage used is needed
import scipy.sparse.linalg      # See above - if these are not imported the code will not run on some Windows 11 HPCs
import logging                  # Used for logging and the logger for code
import multiprocessing          # Used for multithreading when running multiple sims at once

#### Import other files ####

from . import plotting_6D
from . import params_generator
from . import EB_calc

#### Add global variables and logging ####

# Remove all handlers associated with the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Add logger for INFO priority for detailed descriptions of code operations
#     Note that matplotlib tends to spit out 100s of kB of nonsense at the debug
#     level, so only change this if you have some issue with the plots
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("test6D_INFO.log")]
)

#### Define functions ####

# Logging time at a specific event

def give_time(caption):
    logging.info(caption + time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime()))

# Generate mesh positions
        
def make_grids(Nx, Np, x_size, p_size):
    # This function generates the position and momentum grids to integrate
    #   the problem on.
    #
    # Inputs:
    #   Nx      is the number of points in each dimension of position in the mesh
    #   Np      is the number of points in each dimension of momentum in the mesh
    #   x_size  is the physical half-size of the position grid
    #   p_size  is the half-size of the momentum grid
    #
    # Outputs:
    #   grids   is a tuple of the position and momentum grids to be used
    #               for the rectangular elements of the FEM mesh.
    
    # Define a grid of the right size to distort everything else from
    orig_gridx = np.linspace(-1,1,Nx)
    orig_gridp = np.linspace(-1,1,Np)
    
    # Both grids are given as linear based on the position and velocity sizes requested
    grid_x = orig_gridx * x_size
    grid_p = orig_gridp * p_size
    grids = (grid_x, grid_p)
    return grids

def check_simulation_stability(mins):
    # This function makes a qualitative assemsment of simulation stability from
    #   the normalized density minimum.
    #
    # Inputs:
    #   mins    is the density minimum of the simulation normalized to the maximum
    #               density of incoming plasma
    #
    # Outputs:
    #   diag    is the diagnosis of simulation stability level
    
    diag        = 'Stable'
    if mins < -100:
        diag    = 'Extremely unstable'
    elif mins < -10:
        diag    = 'Very unstable'
    elif mins < -1:
        diag    = 'Unstable'
    elif mins < -0.25:
        diag    = 'Dubious'
    elif mins < -0.01:
        diag    = 'Mild local instability, globally stable'
    return diag

def parfor_loop_internals(args):
    # This function calculates the error between an analytical and real solution
    #   for the gyrokinetic example in the paper.
    #   It is used in a parfor loop for the conv_data function to parallelize the
    #   Vlasov simulation, just using the Vlasov equation - no EM feedback.
    #
    # Inputs:
    #   args    is a tuple composed of the real inputs of the function (compressed)
    #               to make them able to be added for parfor:
    #
    #       i           is the current function's place in the queue of simulations to be run
    #       Nx_list     is the list of simulation position-space side lengths that will be used.
    #       Np_list     is the list of simulation momentum-space side lengths that will be used.
    #       params      contains key parameters needed for the simulation
    #
    # Outputs:
    #   l2e     is the RMS error of the simulation, normalized to the scale
    #               of the density function
    #   dof     is the number of degrees of freedom
    
    i, Nx_list, Np_list, params = args                                      # Decompose arguments into the inputs
    grids       = make_grids(Nx_list[i], Np_list[i], 10, 11)                # Make the grids of the current size
    force, stability, result_arrays = eval3D3V(params, grids, 1, 1, False)        # Evaluate Vlasov simulation
    maxe, l2e   = params_generator.value_test(result_arrays, params, 3)     # Find error vs analytical solution
    dof         = 7 * 308 * (Nx_list[i] - 1) ** 3 * (Np_list[i] - 1) ** 3   # Because each hexarract has >= 308 heptapetons inside,
                                                                            # which each have 7 DoF (one for each vertex)                                                                
    return l2e, dof # Returns the RMS error and degrees of freedom (DoF)

def conv_data(Nx_list, Np_list, params, procs):
    # This function calculates the convergence plot between an analytical and real solution
    #   for the shock example in the paper.
    #   It to parallelizes the Vlasov simulation, and does not use EM feedback.
    #
    # Inputs:
    #   Nx_list     is the list of simulation position-space side lengths that will be used.
    #   Np_list     is the list of simulation momentum-space side lengths that will be used.
    #   params      contains key parameters needed for the simulation
    #   procs       is the number of parallel processes that will be used here.
    #
    # Outputs:
    #   plots       Convergence plots
    
    N                   = len(Nx_list)
    
    # Initialise parallel processing to get the RMS error for each simulation
    l2_list             = []
    dof_list            = []
    pool                = multiprocessing.Pool(procs) # Initialize parallelization
    args                = []
    
    # Arguments for parallel for loop initialized as their own for loop
    for i in range(N):
        args.append((i, Nx_list, Np_list, params))
        
    # Parallel for loop
    l2_list, dof_list   = zip(*pool.map(parfor_loop_internals, args))
    
    # Plotting
    plotting_6D.plot_conv([[dof_list, l2_list]], ['Convergence'])

def parfor_loop_internals_em_feedback(args):
    # This function calculates the error between an analytical and real solution
    #   for the shock example in the paper.
    #   It is used in a parfor loop for the conv_data function to parallelize the
    #   Vlasov simulation, but also uses EM feedback.
    #
    # Inputs:
    #   args    is a tuple composed of the real inputs of the function (compressed)
    #               to make them able to be added for parfor:
    #
    #       i           is the current function's place in the queue of simulations to be run
    #       Nx_list     is the list of simulation position-space side lengths that will be used.
    #       Np_list     is the list of simulation momentum-space side lengths that will be used.
    #       params      contains key parameters needed for the simulation
    #       fluids      is a structure containing two objects - the mass list and
    #                       the charge list, which contain the masses and charges of the plasma fluids,
    #                       BOTH IN ORDER AND SPECIFICALLY IN UNITS OF m_Nitrosonium (around 30 amu)!
    #                       DO NOT USE AMU! PROTONS HAVE A MASS OF 1/30, NOT 1!!!
    #                   This is due to a design decision early on, when the code
    #                       was originally meant to be specific to LEO. It works
    #                       fine in other plasmas now, but the units from LEO are still there.
    #
    # Outputs:
    #   l2e     is the RMS error of the simulation, normalized to the scale
    #               of the density function
    #   dof     is the number of degrees of freedom
    
    i, Nx_list, Np_list, params, fluids = args                              # Decompose arguments into the inputs
    grids       = make_grids(Nx_list[i], Np_list[i], 10, 11)                # Make the grids of the current size
    force, stability, result_arrays, params2 = iterateEB_until_result(params, grids, fluids) # Evaluate Vlasov simulation
    maxe, l2e   = params_generator.value_test(result_arrays, params, 4)     # Find error vs analytical solution
    dof         = 7 * 308 * (Nx_list[i] - 1) ** 3 * (Np_list[i] - 1) ** 3   # Because each hexarract has >= 308 heptapetons inside,
                                                                            # which each have 7 DoF (one for each vertex)  
    return l2e, dof

def conv_data_em_feedback(Nx_list, Np_list, params, procs, fluids):
    # This function calculates the convergence plot between an analytical and real solution
    #   for the shock example in the paper.
    #   It to parallelizes the Vlasov simulation, and uses EM feedback.
    #
    # Inputs:
    #   Nx_list     is the list of simulation position-space side lengths that will be used.
    #   Np_list     is the list of simulation momentum-space side lengths that will be used.
    #   params      contains key parameters needed for the simulation
    #   procs       is the number of parallel processes that will be used here.
    #   fluids      is a structure containing two objects - the mass list and
    #                   the charge list, which contain the masses and charges of the plasma fluids,
    #                   BOTH IN ORDER AND SPECIFICALLY IN UNITS OF m_Nitrosonium (around 30 amu)!
    #                   DO NOT USE AMU! PROTONS HAVE A MASS OF 1/30, NOT 1!!!
    #               This is due to a design decision early on, when the code
    #                   was originally meant to be specific to LEO. It works
    #                   fine in other plasmas now, but the units from LEO are still there.
    #
    # Outputs:
    #   plots       Convergence plots
    
    N                   = len(Nx_list)
    
    # Initialise parallel processing to get the RMS error for each simulation
    l2_list             = []
    dof_list            = []
    pool                = multiprocessing.Pool(procs) # Initialize parallelization
    args                = []
    
    # Arguments for parallel for loop initialized as their own for loop
    for i in range(N):
        args.append((i, Nx_list, Np_list, params, fluids))
        
    # Parallel for loop
    l2_list, dof_list   = zip(*pool.map(parfor_loop_internals_em_feedback, args))
    
    # Plotting
    plotting_6D.plot_conv([[dof_list, l2_list]], ['Convergence'])

#### Main calculation function ####

def eval3D3V(params, grids, mass, q, plot = True):
    # This function calculates the forces induced to a 3D volume of plasma by electromagnetic
    #   fields applied to it usinf the Vlasov eqaution. It generates a force, density
    #   plots, and 
    #
    # Inputs:
    #   params          contains key parameters needed for the simulation
    #   grids           is a tuple of the position and momentum grids to be used
    #                       for the rectangular elements of the FEM mesh.
    #   plot            generates plots for the results if true
    #
    # Outputs:
    #   1. Force applied to the electromagnetic fields by the plasma in Newtons
    #   2. Simulation stability parameters to see if anything unstable is happening,
    #       defined by finding the normalized extremeties of the density function.
    #   3. result_arrays - a tuple of the coordinates and plasma component density
    #       at the FEM evaluation nodes, arranged as (f, x, y, z, u, v, w), with u, v, w
    #       being the plasma momentum coordinates
    
    ##  Record starting time and basic information in log file
    logging.info('\n\n\nScript run: ' + __file__)
    give_time('Starting Vlasov simulation, current time = ')
    start           = time.time() # Record starting time for profiling and code time measurement later
    
    # Initialize problem by converting inputs to a new coordinate system:
    #   Units are m, ms, m_NO (30 amu), and e - the mmm unit system
    #   1 T = 3216.178 m_NO/(ms * e)
    #   1 V = 3.216178 m_NO * m^2/(ms^2 * e)
    # Because of this, numerical values for velocity and momentum are the same
    #   for ions in this system of units.
    
    # Extract dimensions from the grids tuple
    grid_x, grid_p  = grids
    
    # Set FEM method order and dimension
    order           = '1'
    dims            = 6
    
    #Build GetFEM mesh using grids
    m               = gf.Mesh('regular simplices', grid_x, grid_x, grid_x, grid_p + params['v_x'], grid_p + params['v_y'], grid_p)
    
    # create a MeshFem of for a field of dimension 1 (i.e. a scalar field, corresponding to f)
    mf              = gf.MeshFem(m, 1)
    
    # assign the Q2 fem to all convexes of the MeshFem
    mf.set_fem(gf.Fem('FEM_PK(' + str(dims) + ',' + order + ')'))
    
    # view the expression of its basis functions on the reference convex
    logging.info('Basis functions per element: ' + str(gf.Fem('FEM_PK(' + str(dims) + ',' + order + ')').poly_str()))
    
    # an exact integration will be used
    mim             = gf.MeshIm(m, gf.Integ('IM_NC(' + str(dims) + ',' + order + ')'))
    
    # detect the borders of the mesh on all 12 sides of the hypercube
    fb0             = m.outer_faces_with_direction([ 1., 0., 0., 0., 0., 0.], 0.01)
    fb1             = m.outer_faces_with_direction([-1., 0., 0., 0., 0., 0.], 0.01)
    fb2             = m.outer_faces_with_direction([0.,  1., 0., 0., 0., 0.], 0.01)
    fb3             = m.outer_faces_with_direction([0., -1., 0., 0., 0., 0.], 0.01)
    fb4             = m.outer_faces_with_direction([0., 0.,  1., 0., 0., 0.], 0.01)
    fb5             = m.outer_faces_with_direction([0., 0., -1., 0., 0., 0.], 0.01)
    fb6             = m.outer_faces_with_direction([0., 0., 0.,  1., 0., 0.], 0.01)
    fb7             = m.outer_faces_with_direction([0., 0., 0., -1., 0., 0.], 0.01)
    fb8             = m.outer_faces_with_direction([0., 0., 0., 0.,  1., 0.], 0.01)
    fb9             = m.outer_faces_with_direction([0., 0., 0., 0., -1., 0.], 0.01)
    fb10            = m.outer_faces_with_direction([0., 0., 0., 0., 0.,  1.], 0.01)
    fb11            = m.outer_faces_with_direction([0., 0., 0., 0., 0., -1.], 0.01)
    
    # mark and label the 12 boundaries
    m.set_region(40, fb0)# Face the plasma is entering on
    m.set_region(41, fb1)# Face the plasma is exiting
    m.set_region(42, fb2)# Side face
    m.set_region(43, fb3)# Side face
    m.set_region(44, fb4)# Side face
    m.set_region(45, fb5)# Side face
    m.set_region(46, fb6)# Velocity extreme
    m.set_region(47, fb7)# Velocity extreme
    m.set_region(48, fb8)# Velocity extreme
    m.set_region(49, fb9)# Velocity extreme
    m.set_region(50, fb10)# Velocity extreme
    m.set_region(51, fb11)# Velocity extreme
    
    # create an empty real model
    md              = gf.Model('real')
    
    # declare that "f" is an unknown of the system, representing density in
    #    particles / (m_NO^3 * m^6 / ms^3) on the finite element method `mf`
    md.add_fem_variable('f', mf)
    
    # Define electric and magnetic fields - this varies based on problem.
    if params['E&B fields included']: # Add both applied E & B fields and fields generated by plasma
        md.add_initialized_fem_data('E1', mf, mf.eval(params['E1'] + ' + params["fE1"](np.transpose(np.array([[x],[y],[z]])))', {**locals(), **globals()}))
        md.add_initialized_fem_data('E2', mf, mf.eval(params['E2'] + ' + params["fE2"](np.transpose(np.array([[x],[y],[z]])))', {**locals(), **globals()}))
        md.add_initialized_fem_data('E3', mf, mf.eval(params['E3'] + ' + params["fE3"](np.transpose(np.array([[x],[y],[z]])))', {**locals(), **globals()}))
        md.add_initialized_fem_data('B1', mf, mf.eval(params['B1'] + ' + params["fB1"](np.transpose(np.array([[x],[y],[z]])))', {**locals(), **globals()}))
        md.add_initialized_fem_data('B2', mf, mf.eval(params['B2'] + ' + params["fB2"](np.transpose(np.array([[x],[y],[z]])))', {**locals(), **globals()}))
        md.add_initialized_fem_data('B3', mf, mf.eval(params['B3'] + ' + params["fB3"](np.transpose(np.array([[x],[y],[z]])))', {**locals(), **globals()}))
        logging.info('Add both applied E & B fields and fields generated by plasma')
    else: # Add only applied E & B fields, not fields from plasma
        md.add_initialized_fem_data('E1', mf, mf.eval(params['E1'], {**locals(), **globals()}))
        md.add_initialized_fem_data('E2', mf, mf.eval(params['E2'], {**locals(), **globals()}))
        md.add_initialized_fem_data('E3', mf, mf.eval(params['E3'], {**locals(), **globals()}))
        md.add_initialized_fem_data('B1', mf, mf.eval(params['B1'], {**locals(), **globals()}))
        md.add_initialized_fem_data('B2', mf, mf.eval(params['B2'], {**locals(), **globals()}))
        md.add_initialized_fem_data('B3', mf, mf.eval(params['B3'], {**locals(), **globals()}))
        logging.info('Add only applied E & B fields, not fields from plasma')
     
    # Define the 6D Vlasov equation
    #   6 dimensions are: x1, x2, x3, p1, p2, p3, in order. x is position, and p is momentum.
    vlasov          = '([X(4); X(5); X(6); 0; 0; 0]*f/' + str(mass) + ' + ' + str(q) + '*([0; 0; 0; E1; E2; E3] + [0; 0; 0; X(5) * B3 - X(6) * B2; X(6) * B1 - X(4) * B3; X(4) * B2 - X(5) * B1]/' + str(mass) + ')*f).Grad_Test_f'
    
    # add generic assembly brick for the bulk of the simulation
    md.add_linear_term(mim, vlasov)
    
    # Use linear terms with multiplier preconditioning for diriclet BCs in position space setting f = DirichletData on the edges
    #   Note that the momentum-space boundary conditions must be Neumann, as their fluxes are assumed to be zero by FEM by default.
    for i in params['BCs']:
        if i[1]     == 'Dirichlet':
            md.add_initialized_fem_data('DirichletData' + str(i[0]), mf, mf.eval(i[2], {**locals(), **globals()}))
            md.add_filtered_fem_variable("mult" + str(i[0]), mf, i[0])
            md.add_linear_term(mim, 'mult' + str(i[0]) + ' * (DirichletData' + str(i[0]) + ' - f)', i[0])
        elif i[1]   == 'Neumann':
            md.add_source_term_brick(mim, 'f', i[2], i[0])
            
    # List variables
    md.variable_list()
    
    # Build all terms in getFEM
    md.assembly("build_all")
    md.assembly("build_rhs")
    logging.info('All terms built')
    
    # Extract rhs and matrix to use scipy for solving
    rhs             = md.rhs()
    K               = md.tangent_matrix()
    
    # Scipy matrix transformation and basic diagnostics
    K               = scipy.sparse.csc_matrix((K.csc_val(),*(K.csc_ind()[::-1])))
    logging.info('Stiffness matrix nonzero values: ' + str(K.getnnz()))
    logging.info('Stiffness matrix size: ' + str(K.shape))
    
    # Scipy matrix solving
    K_shifted       = K + 1e-10 * scipy.sparse.identity(K.shape[0], format='csc') # Diagonal shift to add stability
    ilu             = scipy.sparse.linalg.spilu(K_shifted, drop_tol=1e-4, fill_factor=10) # Adding ilu preconditioner
    M               = scipy.sparse.linalg.LinearOperator(K.shape, matvec=ilu.solve) # Creating preconditioning matrix
    solution, exit_code = scipy.sparse.linalg.bicgstab(K_shifted, rhs, M=M, atol=1e-8) # Matrix solving
    logging.info('SciPy solve, exit code ' + str(exit_code)) # Return exit code (should be 0)
    solution        = np.array(solution) # Extract solution
    logging.info('Solution converted to numpy')
    
    # Grab solution and insert into getFEM
    md.to_variables(solution) # insert back into getFEM for integration
    logging.info('Solution injected to getFEM')
    
    # extracted solution variables
    density         = md.variable('f')
    x               = mf.eval("x")
    y               = mf.eval("y")
    z               = mf.eval("z")
    u               = mf.eval("u")
    v               = mf.eval("v")
    w               = mf.eval("w")
    
    result_arrays   = (density, x, y, z, u, v, w)
    
    #### Boundary integration ####
    
    #Boundaries that connect to external space: 41, 42, 43, 44
    
    x1force         = gf.asm_generic(mim, 0, '[X(4); X(5); X(6)] * X(4) * f', 40, model = md)
    x9force         = gf.asm_generic(mim, 0, '-1 * [X(4); X(5); X(6)] * X(4) * f', 41, model = md)
    y1force         = gf.asm_generic(mim, 0, '[X(4); X(5); X(6)] * X(5) * f', 42, model = md)
    y9force         = gf.asm_generic(mim, 0, '-1 * [X(4); X(5); X(6)] * X(5) * f', 43, model = md)
    z1force         = gf.asm_generic(mim, 0, '[X(4); X(5); X(6)] * X(6) * f', 44, model = md)
    z9force         = gf.asm_generic(mim, 0, '-1 * [X(4); X(5); X(6)] * X(6) * f', 45, model = md)
    
    total_force     = x1force + x9force + y1force + y9force + z1force + z9force
    
    #### Density extraction as a function of position and momentum ####
    
    x_uniq, x_inv   = np.unique(x, return_inverse = True)
    y_uniq, y_inv   = np.unique(y, return_inverse = True)
    z_uniq, z_inv   = np.unique(z, return_inverse = True)
    u_uniq, u_inv   = np.unique(u, return_inverse = True)
    v_uniq, v_inv   = np.unique(v, return_inverse = True)
    w_uniq, w_inv   = np.unique(w, return_inverse = True)
    logging.info('x unique elements: ' + str(x_uniq))
    logging.info('y unique elements: ' + str(y_uniq))
    logging.info('z unique elements: ' + str(z_uniq))
    
    #### Plotting ####
    
    if plot:
        plotting_6D.plot_slice_density_maps(grid_x, grid_p, density, x_inv, y_inv, z_inv, u_inv, v_inv, w_inv, params['v_x'], params['v_y'])
    
    #### Finishing ####
    
    # Print results
    logging.info('Forces in +x (m_NO*m/ms^2): '+str( x1force))
    logging.info('Forces in -x (m_NO*m/ms^2): '+str( x9force))
    logging.info('Forces in +y (m_NO*m/ms^2): '+str( y1force))
    logging.info('Forces in -y (m_NO*m/ms^2): '+str( y9force))
    logging.info('Forces in +z (m_NO*m/ms^2): '+str( z1force))
    logging.info('Forces in -z (m_NO*m/ms^2): '+str( z9force))
    logging.info('Total force (m_NO*m/ms^2): '+str( total_force))
    
    # Log stability result
    logging.info('Simulation is ' + check_simulation_stability(min(density) / params['p_0']))
    logging.info('Stability is ' + str(min(density) / params['p_0']))
    
    # Because each hexarract has >= 308 heptapetons inside, which each have 7 DoF (one for each vertex)
    logging.info('Total degrees of freedom >= ' + str(7 * 308 * (len(grid_x) - 1) ** 3 * (len(grid_p) - 1) ** 3))
    
    logging.info('Time = '+str( time.time() - start) + '\n')
    return total_force*4.9816*(10**-26), [max(density) / params['p_0'], min(density) / params['p_0']], result_arrays

def iterateEB(params, grids, mass, q):
    # This function iterates a Vlasov simulation once for all fluids given, giving
    #   the total EM fields generated, as well as the total force applied by the particles.
    #   The results can be used for the next iteration of the function.
    #
    # Inputs:
    #   params          contains key parameters needed for the simulation
    #   grids           is a tuple of the position and momentum grids to be used
    #                       for the rectangular elements of the FEM mesh.
    #   mass            is an array containing the masses of the plasma fluids,
    #                       BOTH IN ORDER AND SPECIFICALLY IN UNITS OF m_Nitrosonium (around 30 amu)!
    #                       DO NOT USE AMU! PROTONS HAVE A MASS OF 1/30, NOT 1!!!
    #                   This is due to a design decision early on, when the code
    #                       was originally meant to be specific to LEO. It works
    #                       fine in other plasmas now, but the units from LEO are still there.
    #   q               is an array containing the charges of the plasma fluids in
    #                       the same order as the mass array.
    #
    # Outputs:
    #   1. Force applied to the electromagnetic fields by the plasma in Newtons
    #   2. Simulation stability parameters to see if anything unstable is happening,
    #       defined by finding the normalized extremeties of the density function.
    #   3. result_arrays - a tuple of the coordinates and plasma component density
    #       at the FEM evaluation nodes, arranged as (f, x, y, z, u, v, w), with u, v, w
    #       being the plasma momenta coordinates
    #   4. params - contains key parameters needed for the simulation, now updated with
    #       the new electromagnetic fields generated by the plasma.
    
    # Define the E & B field variables, before they are calculated at each point.
    #   While they start as scalars, vectors of the size of the number of FEM mesh
    #   points (not total DoFs - just the mesh points, which are ~2000x lower) are added.
    #   This means that they are vectors for the majority of the code, and are usually
    #   used as such.
    E1tot           = 0
    E2tot           = 0
    E3tot           = 0
    B1tot           = 0
    B2tot           = 0
    B3tot           = 0
    
    # Define the total force, for calculation and usage later. This is an output,
    #   and eventually falls out of the iterateEB_until_result function
    forcetot        = 0
    
    # This for loop iterates over every fluid in the plasma, calculating its density
    #   from the Vlasov equation (through the eval3D3V function) and using it to
    #   get its contribution to both the force and the electromagnetic fields
    for i in range(len(mass)):
        # Find force, sim stability, and densities from each fluid
        force, stability, result_arrays = eval3D3V(params, grids, mass[i], q[i])
        
        # Calculate electromagnetic field from densities
        E1, E2, E3, B1, B2, B3, uniqs   = EB_calc.EB_compute(result_arrays, q[i], grids, False)
        
        # Add up force and EM field contributions for each fluid.
        E1tot       += E1
        E2tot       += E2
        E3tot       += E3
        B1tot       += B1
        B2tot       += B2
        B3tot       += B3
        forcetot    += force
        
    # Define SciPy interpolations for the E & B field values at the gridpoints.
    #   This allows for the GetFEM interface to treat them as continuous fields
    #   over the FEM mesh on the next simulation iteration. This also allows the
    #   E & B fields to be extracted and plotted from the params structure (which
    #   is given as an output).
    # These are added to the params structure to later put into the main eval3D3V
    #   loop and the GetFEM interface.
    params['fE1']   = scipy.interpolate.RegularGridInterpolator(uniqs, E1tot, method='linear')
    params['fE2']   = scipy.interpolate.RegularGridInterpolator(uniqs, E2tot, method='linear')
    params['fE3']   = scipy.interpolate.RegularGridInterpolator(uniqs, E3tot, method='linear')
    params['fB1']   = scipy.interpolate.RegularGridInterpolator(uniqs, B1tot, method='linear')
    params['fB2']   = scipy.interpolate.RegularGridInterpolator(uniqs, B2tot, method='linear')
    params['fB3']   = scipy.interpolate.RegularGridInterpolator(uniqs, B3tot, method='linear')
    
    # Using params as the communication channel, let the code know that the EM fields
    #   are now included.
    params['E&B fields included'] = True
    
    return forcetot, stability, result_arrays, params

def iterateEB_until_result(params, grids, fluids, rmserrormax = 1e-6):
    # This function iterates a Vlasov simulation for all fluids given, giving
    #   the total EM fields generated, as well as the total force applied by the particles.
    #   The results can be used for the next iteration of the function.
    #   It iterates until the RMS error is less than rmserrormax.
    #
    # Inputs:
    #   params          contains key parameters needed for the simulation
    #   grids           is a tuple of the position and momentum grids to be used
    #                       for the rectangular elements of the FEM mesh.
    #   fluids          is a structure containing two objects - the mass list and
    #                       the charge list, which contain the masses and charges of the plasma fluids,
    #                       BOTH IN ORDER AND SPECIFICALLY IN UNITS OF m_Nitrosonium (around 30 amu)!
    #                       DO NOT USE AMU! PROTONS HAVE A MASS OF 1/30, NOT 1!!!
    #                   This is due to a design decision early on, when the code
    #                       was originally meant to be specific to LEO. It works
    #                       fine in other plasmas now, but the units from LEO are still there.
    #   rmserrormax     is the maximum allowable rms error of the current iteration
    #                       and the previous one. Once error drops to this level,
    #                       the simulation is terminated.
    #
    # Outputs:
    #   1. Force applied to the electromagnetic fields by the plasma in Newtons
    #   2. Simulation stability parameters to see if anything unstable is happening,
    #       defined by finding the norm♦alized extremeties of the density function.
    #   3. result_arrays - a tuple of the coordinates and plasma component density
    #       at the FEM evaluation nodes, arranged as (f, x, y, z, u, v, w), with u, v, w
    #       being the plasma momenta coordinates
    #   4. params - contains key parameters needed for the simulation, now updated with
    #       the new electromagnetic fields generated by the plasma.
    
    # Perform an initial iteration to get a baseline of the EM fields and density.
    force, stability, result_arrays, params = iterateEB(params, grids, fluids[0], fluids[1])
    
    # Initialize loop to be used during the simulation
    prev_sim        = result_arrays[0] # Previous simulation results for comparison
    N               = len(prev_sim) # Number of FEM nodes (not DoF, but nodes in the simulation)
    error           = 1 # Initial error estimate is 100% - must come down to converge.
                        # As a side note, technically if you set rmserrormax > 1 then
                        #   only one iteration would happen.
    
    # Main while loop governing the code iteration
    while error     > rmserrormax:
        # Perform one iteration
        force, stability, result_arrays, params = iterateEB(params, grids, fluids[0], fluids[1])
        
        now_sim     = result_arrays[0] # Update current simulation for error calculation
        error       = np.linalg.norm(prev_sim - now_sim) / np.sqrt(N) / np.linalg.norm(prev_sim) # Calculate error
        logging.info('error: '+str( error)) # Log error for this iteration
        prev_sim    = now_sim # Update previous simulation for error calculation
        
    # Output forces and log the final results
    logging.info('Total force (N): '+str( np.linalg.norm(force)))
    logging.info('Total force (kg*m/s/yr): '+str( np.linalg.norm(force*31557600))) # Multiplied by # of seconds in a year
    return force, stability, result_arrays, params