# -*- coding: utf-8 -*-
"""
This file contains functions to mess with the grid, and to create conformal meshes
    for Vlasov plasma simulation. It inputs and outputs a numpy array that is the
    m.pts() of a GetFEM mesh m - that is re-added with m.set_pts(). This is
    for the journal article "OpenVlasov6: Collisionless Plasma Momentum Transfer"
    , by E. Comstock & A. Romero-Calvo.

@author: Eric A. Comstock

v1.3, Eric A. Comstock, 10-Mar-2026
"""

import numpy as np

def nothing(in_pts, custom_bounds = False):
    # This function is the default - it has no effect when applied.
    #
    # Inputs:
    #   in_pts  is the initial set of mesh point, from the grids.
    #
    # Outputs:
    #   out_pts is the return set of mesh points
    
    out_pts = in_pts    # Does nothing - this function is the default.
    return out_pts

def corners_concentrate(in_pts, custom_bounds = False):
    # This function concentrates the points in the corners
    #
    # Inputs:
    #   in_pts  is the initial set of mesh point, from the grids.
    #
    # Outputs:
    #   out_pts is the return set of mesh points
    
    # Creates the out array in the same shape as the in array
    out_pts = in_pts.copy() * 1.0                                   # Make sure to convert to float
    radius  = np.sqrt(in_pts[0]**2 + in_pts[1]**2 + in_pts[2]**2)   # Find position radius to determine how far to extend each point
    if custom_bounds:
        max_r = np.sqrt(max(abs(custom_bounds[0][0]),abs(custom_bounds[0][-1])) ** 2 +
                        max(abs(custom_bounds[1][0]),abs(custom_bounds[1][-1])) ** 2 +
                        max(abs(custom_bounds[2][0]),abs(custom_bounds[2][-1])) ** 2)
    else:
        max_r = np.max(radius)                                          # Maximum position radius
    if max_r == 0:
        max_r = 1
    rel_r = radius / max_r                                          # Relative radius, so that the function's effects do not change based on grid size
    out_pts[3:] = in_pts[3:] * (1 + 1 * rel_r ** 2 / 4)             # Modified points to extend in velocity space at higher radii
    return out_pts

def inlet(in_pts, custom_bounds = False):
    # This function makes positive x narrower in y and z, like an inlet
    #
    # Inputs:
    #   in_pts  is the initial set of mesh point, from the grids.
    #
    # Outputs:
    #   out_pts is the return set of mesh points
    
    # Creates the out array in the same shape as the in array
    out_pts = in_pts.copy() * 1.0
    x = in_pts[0]
    if custom_bounds:
        max_x = custom_bounds[0][-1]
        min_x = custom_bounds[0][0]
    else:
        max_x = np.max(x)
        min_x = np.min(x)
    if max_x - min_x == 0:
        max_x = min_x + 1
    rel_x = (x - min_x) / (max_x - min_x)
    out_pts[1:3] = in_pts[1:3] * (1 - 0.8 * rel_x)
    return out_pts