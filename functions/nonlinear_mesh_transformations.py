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

def nothing(in_pts):
    # This function is the default - it has no effect when applied.
    #
    # Inputs:
    #   in_pts  is the initial set of mesh point, from the grids.
    #
    # Outputs:
    #   out_pts is the return set of mesh points
    
    out_pts = in_pts    # Does nothing - this function is the default.
    return out_pts

def corners_concentrate(in_pts):
    # This function concentrates the points in the corners
    #
    # Inputs:
    #   in_pts  is the initial set of mesh point, from the grids.
    #
    # Outputs:
    #   out_pts is the return set of mesh points
    
    # Creates the out array in the same shape as the in array
    out_pts = in_pts.copy()
    radius  = np.sqrt(in_pts[0]**2 + in_pts[1]**2 + in_pts[2]**2)
    max_r = np.max(radius)
    rel_r = radius / max_r
    out_pts[3:5] = in_pts[3:5] * (1 + 1 * rel_r ** 2 / 4)  
    #print(out_pts.shape)
    return out_pts