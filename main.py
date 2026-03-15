# -*- coding: utf-8 -*-
"""
This file is the main file needed for OpenVlasov6. Call other functions from here.
    This file is described in the journal article "OpenVlasov6: A 3D-3V Fully Kinetic
    Multifluid Vlasov Solver" in Physics of Plasmas, by E. Comstock & A. Romero-Calvo.
To run the code under set parameters, modify the part of the code that says "Modify this part!"

@author: Eric A. Comstock

v1.3.2, Eric A. Comstock, 15-Mar-2026
v1.3.1, Eric A. Comstock, 13-Mar-2026
v1.3, Eric A. Comstock, 10-Mar-2026
v1.2.3, Eric A. Comstock, 9-Mar-2026
v1.2.2, Eric A. Comstock, 23-Feb-2026
v1.2.1, Eric A. Comstock, 10-Feb-2026
v1.2, Eric A. Comstock, 3-Feb-2026
v1.1, Eric A. Comstock, 20-Nov-2025
v1.0.1, Eric A. Comstock, 14-Oct-2025
v1.0, Eric A. Comstock, 3-Oct-2025
v0.0, Eric A. Comstock, 2-Oct-2025
"""

#### Import basic modules ####

import numpy as np              # Used for vector algebra

#### Import other files ####

from functions import plotting_6D
from functions import params_generator
from functions import EB_calc
from functions import Vlasov_testing_code_6D
from functions import utils

#### Running code with specifics - Modify this part! ####

# Sample code for your initial run of a plasma in an EM field - feel free to delete this
grids2  = Vlasov_testing_code_6D.make_grids_sinh(4, 4, 10, 11, 1e-4, 2) # Rough representation of nonuniformity in position and momentum space
force, stability, result_arrays = Vlasov_testing_code_6D.eval3D3V(params_generator.params_example2(), grids2, 1, 1)  # Test case 2

#### Shelving all data for potential later use ####

utils.save_everything()