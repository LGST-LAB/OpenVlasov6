# -*- coding: utf-8 -*-
"""
This file is the main file needed for OpenVlasov6. Call other functions from here.
    This file is described in the journal article "OpenVlasov6: A 3D-3V Fully Kinetic
    Multifluid Vlasov Solver" in Computer Physics Communications, by E. A. Comstock 
    & K. Poulios & A. Romero-Calvo.
To run the code under set parameters, modify the part of the code that says "Modify this part!"

@author: Eric A. Comstock

1.4.1, Eric A. Comstock, 10-May-2026
1.4.0, Eric A. Comstock, 30-Apr-2026
1.4.0-rc.2, Eric A. Comstock, 23-Apr-2026
1.4.0-rc.1, Eric A. Comstock, 17-Apr-2026
1.4.0-hex.4, Eric A. Comstock, 16-Apr-2026
1.4.0-hex.3, Eric A. Comstock, 13-Apr-2026
1.4.0-hex.2, Eric A. Comstock, 12-Apr-2026
1.4.0-hex.1, Konstantinos Poulios, 9-Apr-2026
1.4.0-dev.1, Eric A. Comstock, 9-Apr-2026
1.3.5-dev.1, Eric A. Comstock, 24-Mar-2026
1.3.4, Eric A. Comstock, 19-Mar-2026
1.3.3, Eric A. Comstock, 17-Mar-2026
1.3.2, Eric A. Comstock, 15-Mar-2026
1.3.1, Eric A. Comstock, 13-Mar-2026
1.3.0, Eric A. Comstock, 10-Mar-2026
1.2.3, Eric A. Comstock, 9-Mar-2026
1.2.2, Eric A. Comstock, 23-Feb-2026
1.2.1, Eric A. Comstock, 10-Feb-2026
1.2.0, Eric A. Comstock, 3-Feb-2026
1.1.0, Eric A. Comstock, 20-Nov-2025
1.0.1, Eric A. Comstock, 14-Oct-2025
1.0.0, Eric A. Comstock, 3-Oct-2025
"""

#### Import basic modules ####

import numpy as np                              # Used for vector algebra

#### Import other files ####

from functions import plotting_6D               # For plotting slices of the 6D space
from functions import params_generator          # To pre-generate sets of parameters to run for simulations
from functions import EB_calc                   # For EM field calculations, if nessesary
from functions import Vlasov_testing_code_6D    # Base of the code
from functions import utils                     # general utility functions

#### Running code with specifics - Modify this part! ####

grids2  = Vlasov_testing_code_6D.make_grids_sinh(4, 4, 10, 11, 1e-4, 2) # Rough representation of nonuniformity in position and momentum space

force, stability, result_arrays = Vlasov_testing_code_6D.eval3D3V(params_generator.params_example2(), grids2, 1, 1)  # Test case 2

#### Shelving all data for potential later use ####

# Fix this later before 1.4 comes out
#utils.save_everything()