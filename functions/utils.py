# -*- coding: utf-8 -*-
"""
This file contains various general utility functions not directly relevant to the
plasma simulation. This is
for the journal article "OpenVlasov6: A 3D-3V Fully Kinetic
Multifluid Vlasov Solver" in Computer Physics Communications, by E. A. Comstock 
& K. Poulios & A. Romero-Calvo.

@author: Eric A. Comstock

1.4.0-rc.2, Eric A. Comstock, 23-Apr-2026
1.3.4, Eric A. Comstock, 19-Mar-2026
1.3.2, Eric A. Comstock, 15-Mar-2026
"""

#### Import basic modules ####

from config.MPI_config import * # Used for controlling configuration parameters
import shelve                   # Used to save data in case it is needed later
import time                     # Used for getting time for logging and file names

#### MPI4py ####

# This function is used to detect if the program is MPI or not, import mpi4py if
#   it is, and error if it is not.
if MPI_toggle:
    try:
        from mpi4py import MPI
    except:
        raise Exception("MPI not installed on this device. Either install MPI4py, or disable MPI using config/MPI_config.py by setting MPI_toggle to False.")
    
    # Only initialize MPI variables if MPI is enabled in config
    comm = MPI.COMM_WORLD   # Initializing communciation system for MPI
    rank = comm.Get_rank()  # Detect rank of current processor
    size = comm.Get_size()  # Detect total number of ranks
    
#### Functions ####
    
def MPI_toggle_decorator(function):
    # This function decorates an input function to activate either only on the main
    #   code or at rank == 0 for MPI
    #
    # Inputs:
    #   function    is the input function that we want to activate either only on
    #                   the main code or at rank == 0 for MPI.
    #
    # Outputs:
    #   wrapper     is the output function, and is just function, modified so that
    #                   it only activates on the main code or at rank == 0 for MPI.
    
    def wrapper(*args, **kwargs):
        # This function is the modified version of the original function, modified
        #   so that it only activates on the main code or at rank == 0 for MPI.
        #
        # Inputs:
        #   *args       is the normal function arguments
        #   **kwargs    is any extra arguments
        #
        # Outputs:
        #   results     is whatever the function is supposed to output
        
        # Only activate function if MPI is off or rank == 0
        if MPI_toggle:
            if rank == 0:
                result = function(*args, **kwargs)
        else:
            result = function(*args, **kwargs)
        
        return result
    return wrapper

@MPI_toggle_decorator # Add decorator to ensure that save_everything is only used when it is truly needed.
def save_everything():
    # This saves the whole namespace
    #
    # Inputs: None
    #
    # Outputs: None
    
    # Generate unique filename based on time
    filename = str(time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())) + 'shelve.out'
    
    # Open the shelf
    my_shelf = shelve.open(filename,'n')
    
    # Save all variables in globals()
    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    
    # Close the shelf so it can be used later by other programs
    my_shelf.close()