#########################################################################################
# Copyright 2023-2024 Lawrence Livermore National Security, LLC and other 
# project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# written by: Kyle Champley <champley@gmail.com>, <champley1@llnl.gov>
# project PI: Brian Maddox
#
# SymmetricProjectors provides Numba-accelerated Python routines for
# forward and back projection and analytic inversion algorithms of
# parallel-beam and cone-beam cylindrically anti-symmetric objects
# with tilted axis of symmetry (similar, but more general Abel Transforms).
# This exact some functionality is already implemented in LEAP using C++/CUDA.
# Although the C++ and CUDA implementation are faster, this file provides
# a pure python implementation which may be more advantageous for some users.
#########################################################################################

# CONE-BEAM COORDINATE DESCRIPTION
# Definition of the origin of the coordinate system, i.e., the (0,0,0) point:
#    the z-coordinate of the source is defined to be the z=0 plane
#    the location of the origin in (x,y) coordinates is the axis of symmetry of the object
# Source Position:
#    (tau,-sod,0)
# Detector Pixel Positions:
#    (cols[i], sdd-sod, rows[j]), where
#    cols[i] = (i - centerCol)*pixelWidth, for i = 0, 1, ..., numCols-1
#    rows[j] = (j - centerRow)*pixelHeight, for j = 0, 1, ..., numRows-1

from numba import jit, prange
import numpy as np


@jit(nopython=True,parallel=True)
def inverse_transform(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0, sod, tau):
    
    numRows = g.shape[0]
    numCols = g.shape[1]
    numZ = f.shape[0]
    numR = f.shape[1]
    
    cos_beta = np.cos(beta*np.pi / 180.0)
    sin_beta = np.sin(beta*np.pi / 180.0)
    if np.abs(sin_beta) < 1.0e-4:
        sin_beta = 0.0
        cos_beta = 1.0
    tan_beta = sin_beta / cos_beta
    sec_beta = 1.0 / cos_beta

    N_phi = numCols + ((numCols + 1) % 2)
    T_phi = 2.0 * np.pi / float(N_phi)
    R_sq = sod * sod

    scaling = 1.0 / (2.0 * float(N_phi))

    f[:,:] = 0.0
    for j in prange(numR):
        r_val = T_r * j + r_0
        for iphi in range(N_phi):
            phi = float(iphi) * T_phi - 0.5 * np.pi
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            if sod > 0.0: # cone-beam
                for k in range(numZ):
                    z_val = k * T_z + z_0
                    v_denom_inv = 1.0 / (sod + r_val * sin_phi * cos_beta - z_val * sin_beta)
                    s_val = (tau - r_val * cos_phi) * v_denom_inv
                    if r_val * s_val < 0.0:
                        s_val *= -1.0
                    s_arg = (s_val - u_0) / T_u
                    v_arg = ((r_val * sin_phi * sin_beta + z_val * cos_beta) * v_denom_inv - v_0) / T_v
                    theWeight = R_sq * v_denom_inv * v_denom_inv * scaling

                    if s_arg < 0.0 or s_arg > float(numCols - 1):
                        continue

                    s_arg = max(float(0.0), min(s_arg, float(numCols - 1)))
                    s_low = int(s_arg)
                    s_high = min(s_low + 1, numCols - 1)
                    ds = s_arg - float(s_low)

                    v_arg = max(float(0.0), min(v_arg, float(numRows - 1)))
                    v_low = int(v_arg)
                    v_high = min(v_low + 1, numRows - 1)
                    dv = v_arg - float(v_low)

                    f[k,j] += ((1.0 - dv) * ((1.0 - ds) * g[v_low, s_low] + ds * g[v_low, s_high]) + dv * ((1.0 - ds) * g[v_high, s_low] + ds * g[v_high, s_high])) * theWeight
                        
            else: # parallel-beam
                s_val = r_val * cos_phi

                if r_val * s_val < 0.0:
                    s_val *= -1.0

                s_arg = (s_val - u_0) / T_u

                if s_arg < 0.0 or s_arg > float(numCols - 1):
                    continue

                s_arg = max(float(0.0), min(s_arg, float(numCols-1)))
                s_low = int(s_arg)
                s_high = min(s_low + 1, numCols-1)
                ds = s_arg - float(s_low)

                for k in range(numZ):
                    z_val = T_z * k + z_0
                    v_arg = ((r_val * sin_phi * sin_beta + z_val * cos_beta) - v_0) / T_v
                    theWeight = scaling

                    f[k,j] += ((1.0 - ds) * g[k, s_low] + ds* g[k, s_high]) * theWeight

@jit(nopython=True,parallel=True)
def project_cone(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0, sod, tau):
    
    numRows = g.shape[0]
    numCols = g.shape[1]
    numZ = f.shape[0]
    numR = f.shape[1]
    
    cos_beta = np.cos(beta*np.pi / 180.0)
    sin_beta = np.sin(beta*np.pi / 180.0)
    if np.abs(sin_beta) < 1.0e-4:
        sin_beta = 0.0
        cos_beta = 1.0

    #N_r = int(0.5 + 0.5*numR)
    #N_r = int(0.5 - r_0/T_r)
    #N_r = int(np.floor(1.0 - r_0/T_r))
    
    r_center_ind = -r_0/T_r
    N_r_left = int(np.floor(r_center_ind))+1
    N_r_right = numR - N_r_left
    #r_max = max((numR - 1)*T_r + r_0, np.abs(r_0))
    
    Rcos_sq_plus_tau_sq = sod*sod*cos_beta*cos_beta + tau*tau

    for j in prange(numRows):
        v = j * T_v + v_0
        X = cos_beta - v * sin_beta

        z_shift = (-sod*sin_beta - z_0) / T_z
        z_slope = (sin_beta + v * cos_beta) / T_z

        for k in range(numCols):
            u_unbounded = k * T_u + u_0
            u = np.abs(u_unbounded)

            if u_unbounded < 0.0:
                rInd_max = N_r_left
                r_max = np.abs(r_0)
            else:
                rInd_max = N_r_right
                r_max = (numR - 1)*T_r + r_0
            r_min = 0.5*T_r

            sec_sq_plus_u_sq = X * X + u * u
            b_ti = X * sod*cos_beta + u * tau
            a_ti_inv = 1.0 / sec_sq_plus_u_sq
            disc_ti_shift = b_ti * b_ti - sec_sq_plus_u_sq * Rcos_sq_plus_tau_sq # new

            if np.abs(disc_ti_shift) < 1.0e-8:
                disc_ti_shift = 0.0
            if np.abs(sec_sq_plus_u_sq) < 1.0e-8 or disc_ti_shift > 0.0:
                g[j, k] = 0.0
                continue

            rInd_min = int(np.ceil((np.sqrt(-disc_ti_shift / sec_sq_plus_u_sq)) / T_r))
            r_prev = float(rInd_min)*T_r
            disc_sqrt_prev = np.sqrt(disc_ti_shift + r_prev * r_prev*sec_sq_plus_u_sq)

            curVal = 0.0

            # Go back one sample and check
            if rInd_min >= 1:
                r_absoluteMinimum = np.sqrt(-disc_ti_shift / sec_sq_plus_u_sq)
                #rInd_min_minus = max(0, min(N_r - 1, int(np.ceil((r_absoluteMinimum) / T_r - 1.0))))
                rInd_min_minus = max(0, int(np.ceil((r_absoluteMinimum) / T_r - 1.0)))

                if u_unbounded < 0.0:
                    ir_shifted_or_flipped = N_r_left - 1 - rInd_min_minus
                else:
                    ir_shifted_or_flipped = N_r_left + rInd_min_minus
                    
                if r_absoluteMinimum < r_max and disc_sqrt_prev > 0.0 and 0 <= ir_shifted_or_flipped and ir_shifted_or_flipped <= numR-1:
                    iz_arg_low = (b_ti - 0.5*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift
                    if 0.0 <= iz_arg_low and iz_arg_low <= numZ - 1:
                        iz_arg_low_floor = int(iz_arg_low)
                        dz = iz_arg_low - float(iz_arg_low_floor)
                        iz_arg_low_ceil = min(iz_arg_low_floor + 1, numZ - 1)
                        curVal += disc_sqrt_prev * a_ti_inv*((1.0 - dz)*f[iz_arg_low_floor, ir_shifted_or_flipped] + dz * f[iz_arg_low_ceil, ir_shifted_or_flipped])

                    iz_arg_high = (b_ti + 0.5*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift
                    if 0.0 <= iz_arg_high and iz_arg_high <= numZ - 1:
                        iz_arg_high_floor = int(iz_arg_high)
                        dz = iz_arg_high - float(iz_arg_high_floor)
                        iz_arg_high_ceil = min(iz_arg_high_floor + 1, numZ - 1)
                        curVal += disc_sqrt_prev * a_ti_inv*((1.0 - dz)*f[iz_arg_high_floor, ir_shifted_or_flipped] + dz * f[iz_arg_high_ceil, ir_shifted_or_flipped])

            for ir in range(rInd_min, rInd_max):
                if u_unbounded < 0.0:
                    ir_shifted_or_flipped = N_r_left - 1 - ir
                else:
                    ir_shifted_or_flipped = N_r_left + ir

                r_next = r_prev + T_r
                disc_sqrt_next = np.sqrt(disc_ti_shift + r_next * r_next*sec_sq_plus_u_sq)

                if 0 <= ir_shifted_or_flipped and ir_shifted_or_flipped <= numR-1:
                    # Negative t interval
                    # low:  (b_ti - disc_sqrt_next) * a_ti_inv
                    # high: (b_ti - disc_sqrt_prev) * a_ti_inv

                    # Positive t interval
                    # low:  (b_ti + disc_sqrt_prev) * a_ti_inv
                    # high: (b_ti + disc_sqrt_next) * a_ti_inv

                    #(b_ti - disc_sqrt_next) * a_ti_inv + (b_ti - disc_sqrt_prev) * a_ti_inv
                    iz_arg_low = (b_ti - 0.5*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift
                    if 0.0 <= iz_arg_low and iz_arg_low <= numZ - 1:
                        iz_arg_low_floor = int(iz_arg_low)
                        dz = iz_arg_low - float(iz_arg_low_floor)
                        iz_arg_low_ceil = min(iz_arg_low_floor + 1, numZ - 1)

                        curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[iz_arg_low_floor, ir_shifted_or_flipped] + dz * f[iz_arg_low_ceil, ir_shifted_or_flipped])

                    iz_arg_high = (b_ti + 0.5*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift
                    if 0.0 <= iz_arg_high and iz_arg_high <= numZ - 1:
                        iz_arg_high_floor = int(iz_arg_high)
                        dz = iz_arg_high - float(iz_arg_high_floor)
                        iz_arg_high_ceil = min(iz_arg_high_floor + 1, numZ - 1)

                        curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[iz_arg_high_floor, ir_shifted_or_flipped] + dz * f[iz_arg_high_ceil, ir_shifted_or_flipped])

                # update radius and sqrt for t calculation
                r_prev = r_next
                disc_sqrt_prev = disc_sqrt_next
            g[j, k] = curVal * np.sqrt(1.0 + u * u + v * v)


@jit(nopython=True,parallel=True)
def backproject_cone(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0, sod, tau):
    
    numRows = g.shape[0]
    numCols = g.shape[1]
    numZ = f.shape[0]
    numR = f.shape[1]

    cos_beta = np.cos(beta*np.pi / 180.0)
    sin_beta = np.sin(beta*np.pi / 180.0)
    if np.abs(sin_beta) < 1.0e-4:
        sin_beta = 0.0
        cos_beta = 1.0
    tan_beta = sin_beta / cos_beta
    sec_beta = 1.0 / cos_beta

    ind_split = -u_0 / T_u
    if np.abs(ind_split - np.round(ind_split)) < 1.0e-4:
        iu_min_left = 0
        iu_max_left = int(np.round(ind_split))
        iu_min_right = iu_max_left
        iu_max_right = numCols - 1
    else:
        iu_min_left = 0
        iu_max_left = int(ind_split)
        iu_min_right = int(np.ceil(ind_split))
        iu_max_right = numCols - 1

    for j in prange(numR):
        r_unbounded = j * T_r + r_0
        r = np.abs(r_unbounded)
        r_inner = r - 0.5*T_r
        r_outer = r + 0.5*T_r

        if r_unbounded < 0.0:
            # left half
            iu_min = 0
            #iu_max = int(-u_0 / T_u)
            iu_max = iu_max_left
            #print('left: ' + str(iu_min) + ', ' + str(iu_max))
        else:
            # right half
            #iu_min = int(np.ceil(-u_0 / T_u))
            iu_min = iu_min_right
            iu_max = numCols - 1
            #print('right: ' + str(iu_min) + ', ' + str(iu_max))

        disc_shift_inner = (r_inner*r_inner - tau*tau)*sec_beta*sec_beta # r_inner^2
        disc_shift_outer = (r_outer*r_outer - tau*tau)*sec_beta*sec_beta # r_outer^2

        for k in range(numZ):
            z = k * T_z + z_0

            Z = (sod + z * sin_beta)*sec_beta # nominal value: R
            z_slope = (z + sod*sin_beta)*sec_beta # nominal value: 0

            curVal = 0.0
            for iu in range(iu_min, iu_max+1):
                u = np.abs(iu * T_u + u_0)

                disc_outer = u * u*(r_outer*r_outer - Z * Z) + 2.0*Z*sec_beta*tau*u + disc_shift_outer # u^2*(r^2 - R^2) + r^2
                if disc_outer > 0.0:

                    b_ti = Z * sec_beta + tau*u
                    a_ti_inv = 1.0 / (u*u + sec_beta * sec_beta)

                    disc_inner = u * u*(r_inner*r_inner - Z * Z) + 2.0*Z*sec_beta*tau*u + disc_shift_inner # disc_outer > disc_inner
                    if disc_inner > 0.0:
                        disc_inner = np.sqrt(disc_inner)
                        disc_outer = np.sqrt(disc_outer)
                        # first t interval
                        # t interval: (b_ti-sqrt(disc_outer))*a_ti_inv to (b_ti-sqrt(disc_inner))*a_ti_inv
                        t_1st_low = (b_ti - disc_outer)*a_ti_inv
                        t_1st_high = (b_ti - disc_inner)*a_ti_inv
                        v_1st_arg = 2.0*z_slope / (t_1st_low + t_1st_high) - tan_beta

                        theWeight_1st = np.sqrt(1.0 + u * u + v_1st_arg * v_1st_arg + tan_beta * tan_beta) * (t_1st_high - t_1st_low)

                        v_1st_arg = max(0.0, min(float(numRows - 1.001), (v_1st_arg - v_0) / T_v))
                        v_1st_arg_floor = int(v_1st_arg)
                        dv_1st = v_1st_arg - float(v_1st_arg_floor)
                        curVal += theWeight_1st * ((1.0 - dv_1st)*g[v_1st_arg_floor, iu] + dv_1st * g[(v_1st_arg_floor + 1), iu])

                        # second t interval
                        # t interval: (b_ti+sqrt(disc_inner))*a_ti_inv to (b_ti+sqrt(disc_outer))*a_ti_inv
                        t_2nd_low = (b_ti + disc_inner)*a_ti_inv
                        t_2nd_high = (b_ti + disc_outer)*a_ti_inv
                        v_2nd_arg = 2.0*z_slope / (t_2nd_low + t_2nd_high) - tan_beta

                        theWeight_2nd = np.sqrt(1.0 + u * u + v_2nd_arg * v_2nd_arg + tan_beta * tan_beta) * (t_2nd_high - t_2nd_low)

                        v_2nd_arg = max(0.0, min(float(numRows - 1.001), (v_2nd_arg - v_0) / T_v))
                        v_2nd_arg_floor = int(v_2nd_arg)
                        dv_2nd = v_2nd_arg - float(v_2nd_arg_floor)
                        curVal += theWeight_2nd * ((1.0 - dv_2nd)*g[v_2nd_arg_floor, iu] + dv_2nd * g[(v_2nd_arg_floor + 1), iu])
                    else:
                        disc_outer = np.sqrt(disc_outer)
                        # t interval: (b_ti-sqrt(disc_outer))*a_ti_inv to (b_ti+sqrt(disc_outer))*a_ti_inv
                        # t interval midpoint: b_ti*a_ti_inv

                        # take mid value for interval to find iv
                        v_arg = z_slope / (b_ti*a_ti_inv) - tan_beta

                        theWeight = np.sqrt(1.0 + u * u + v_arg * v_arg + tan_beta * tan_beta) * 2.0*disc_outer*a_ti_inv

                        v_arg = max(0.0, min(float(numRows - 1.001), (v_arg - v_0) / T_v))
                        v_arg_floor = int(v_arg)
                        dv = v_arg - float(v_arg_floor)
                        curVal += theWeight * ((1.0 - dv)*g[v_arg_floor, iu] + dv * g[(v_arg_floor + 1), iu])
            f[k,j] = curVal
    

@jit(nopython=True,parallel=True)
def project_parallel(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0):

    numRows = g.shape[0]
    numCols = g.shape[1]
    numZ = f.shape[0]
    numR = f.shape[1]
        
    cos_beta = np.cos(beta*np.pi / 180.0)
    sin_beta = np.sin(beta*np.pi / 180.0)
    if np.abs(sin_beta) < 1.0e-4:
        sin_beta = 0.0
        cos_beta = 1.0
    sec_beta = 1.0 / cos_beta
    
    r_center_ind = -r_0/T_r
    N_r_left = int(np.floor(r_center_ind))+1
    N_r_right = numR - N_r_left

    #r_max = max((numR - 1)*T_r + r_0, np.abs(r_0))

    for j in prange(numRows):
        v = j*T_v+v_0
        Y = sin_beta + v * cos_beta
        X = cos_beta - v * sin_beta

        z_shift = (v*cos_beta - z_0) / T_z
        z_slope = sin_beta / T_z

        for k in range(numCols):
            u_unbounded = k * T_u + u_0
            u = np.abs(u_unbounded)

            if u_unbounded < 0.0:
                rInd_max = N_r_left
                r_max = np.abs(r_0)
            else:
                rInd_max = N_r_right
                r_max = (numR - 1)*T_r + r_0
            r_min = 0.5*T_r

            b_ti = v * sin_beta
            a_ti_inv = sec_beta
            disc_ti_shift = -u * u

            rInd_min = int(np.ceil(u / T_r))
            r_prev = float(rInd_min)*T_r
            disc_sqrt_prev = np.sqrt(max(0.0, disc_ti_shift + r_prev * r_prev))

            curVal = 0.0

            ####################################################################################
            # Go back one sample and check
            if rInd_min >= 1:
                r_absoluteMinimum = u
                
                rInd_min_minus = max(0, int(np.ceil(r_absoluteMinimum / T_r - 1.0)))

                if u_unbounded < 0.0:
                    ir_shifted_or_flipped = N_r_left - 1 - rInd_min_minus
                else:
                    ir_shifted_or_flipped = N_r_left + rInd_min_minus
                
                if r_absoluteMinimum < r_max and 0 <= ir_shifted_or_flipped and ir_shifted_or_flipped <= numR-1:
                    iz_arg_low = (b_ti - 0.5*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift
                    if 0.0 <= iz_arg_low and iz_arg_low <= numZ - 1:
                        iz_arg_low_floor = int(iz_arg_low)
                        dz = iz_arg_low - float(iz_arg_low_floor)
                        iz_arg_low_ceil = min(iz_arg_low_floor + 1, numZ - 1)

                        curVal += max(0.0, disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[iz_arg_low_floor,ir_shifted_or_flipped] + dz * f[iz_arg_low_ceil,ir_shifted_or_flipped])

                    iz_arg_high = (b_ti + 0.5*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift
                    if 0.0 <= iz_arg_high and iz_arg_high <= numZ - 1:
                        iz_arg_high_floor = int(iz_arg_high)
                        dz = iz_arg_high - float(iz_arg_high_floor)
                        iz_arg_high_ceil = min(iz_arg_high_floor + 1, numZ - 1)

                        curVal += max(0.0, disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[iz_arg_high_floor, ir_shifted_or_flipped] + dz * f[iz_arg_high_ceil,ir_shifted_or_flipped])
            ####################################################################################

            for ir in range(rInd_min, rInd_max):
            
                if u_unbounded < 0.0:
                    ir_shifted_or_flipped = N_r_left - 1 - ir
                else:
                    ir_shifted_or_flipped = N_r_left + ir
            
                r_next = r_prev + T_r
                disc_sqrt_next = np.sqrt(disc_ti_shift + r_next * r_next)

                if 0 <= ir_shifted_or_flipped and ir_shifted_or_flipped <= numR-1:
                    # Negative t interval
                    # low:  (b_ti - disc_np.sqrt_next) * a_ti_inv
                    # high: (b_ti - disc_np.sqrt_prev) * a_ti_inv

                    # Positive t interval
                    # low:  (b_ti + disc_np.sqrt_prev) * a_ti_inv
                    # high: (b_ti + disc_np.sqrt_next) * a_ti_inv

                    # b_ti - disc_np.sqrt_next) * a_ti_inv + (b_ti - disc_np.sqrt_prev) * a_ti_inv
                    iz_arg_low = (b_ti - 0.5*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift
                    if 0.0 <= iz_arg_low and iz_arg_low <= numZ - 1:
                        iz_arg_low_floor = int(iz_arg_low)
                        dz = iz_arg_low - float(iz_arg_low_floor)
                        iz_arg_low_ceil = min(iz_arg_low_floor + 1, numZ - 1)

                        curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[iz_arg_low_floor, ir_shifted_or_flipped] + dz * f[iz_arg_low_ceil, ir_shifted_or_flipped])

                    iz_arg_high = (b_ti + 0.5*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift
                    if 0.0 <= iz_arg_high and iz_arg_high <= numZ - 1:
                        iz_arg_high_floor = int(iz_arg_high)
                        dz = iz_arg_high - float(iz_arg_high_floor)
                        iz_arg_high_ceil = min(iz_arg_high_floor + 1, numZ - 1)

                        curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[iz_arg_high_floor, ir_shifted_or_flipped] + dz * f[iz_arg_high_ceil, ir_shifted_or_flipped])

                # update radius and np.sqrt for t calculation
                r_prev = r_next
                disc_sqrt_prev = disc_sqrt_next
            g[j, k] = curVal


@jit(nopython=True,parallel=True)
def backproject_parallel(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0):
    
    numRows = g.shape[0]
    numCols = g.shape[1]
    numZ = f.shape[0]
    numR = f.shape[1]
    
    cos_beta = np.cos(beta*np.pi / 180.0)
    sin_beta = np.sin(beta*np.pi / 180.0)
    if np.abs(sin_beta) < 1.0e-4:
        sin_beta = 0.0
        cos_beta = 1.0
    tan_beta = sin_beta / cos_beta
    sec_beta = 1.0 / cos_beta

    ind_split = -u_0 / T_u
    if np.abs(ind_split - np.round(ind_split)) < 1.0e-4:
        iu_min_left = 0
        iu_max_left = int(np.round(ind_split))
        iu_min_right = iu_max_left
        iu_max_right = numCols - 1
    else:
        iu_min_left = 0
        iu_max_left = int(ind_split)
        iu_min_right = int(np.ceil(ind_split))
        iu_max_right = numCols - 1

    for j in prange(numR):
        r_unbounded = j * T_r + r_0
        r = np.abs(r_unbounded)
        r_inner = r - 0.5*T_r
        r_outer = r + 0.5*T_r

        if r_unbounded < 0.0:
            # left half
            iu_min = 0
            #iu_max = int(-u_0 / T_u)
            iu_max = iu_max_left
        else:
            # right half
            #iu_min = int(np.ceil(-u_0 / T_u))
            iu_min = iu_min_right
            iu_max = numCols - 1

        disc_shift_inner = r_inner * r_inner # r_inner^2
        disc_shift_outer = r_outer * r_outer # r_outer^2

        for k in range(numZ):
            z = k * T_z + z_0

            curVal = 0.0
            for iu in range(iu_min, iu_max+1):
                u = np.abs(iu * T_u + u_0)

                disc_outer = disc_shift_outer - u * u # u^2*(r^2 - R^2) + r^2
                if disc_outer > 0.0:

                    b_ti = z * tan_beta
                    a_ti_inv = cos_beta

                    disc_inner = disc_shift_inner - u * u # disc_outer > disc_inner
                    if disc_inner > 0.0:
                        disc_inner = np.sqrt(disc_inner)
                        disc_outer = np.sqrt(disc_outer)
                        # first t interval
                        # t interval: (b_ti-np.sqrt(disc_outer))*a_ti_inv to (b_ti-np.sqrt(disc_inner))*a_ti_inv
                        t_1st_low = (b_ti - disc_outer)*a_ti_inv
                        t_1st_high = (b_ti - disc_inner)*a_ti_inv
                        v_1st_arg = -0.5*(t_1st_low + t_1st_high)*tan_beta + z * sec_beta

                        v_1st_arg = max(0.0, min(float(numZ - 1.001), (v_1st_arg - z_0) / T_z))
                        v_1st_arg_floor = int(v_1st_arg)
                        dv_1st = v_1st_arg - float(v_1st_arg_floor)
                        curVal += (t_1st_high - t_1st_low)*((1.0 - dv_1st)*g[v_1st_arg_floor, iu] + dv_1st * g[(v_1st_arg_floor + 1), iu])

                        # second t interval
                        # t interval: (b_ti+np.sqrt(disc_inner))*a_ti_inv to (b_ti+np.sqrt(disc_outer))*a_ti_inv
                        t_2nd_low = (b_ti + disc_inner)*a_ti_inv
                        t_2nd_high = (b_ti + disc_outer)*a_ti_inv
                        v_2nd_arg = -0.5*(t_2nd_low + t_2nd_high)*tan_beta + z * sec_beta


                        v_2nd_arg = max(0.0, min(float(numZ - 1.001), (v_2nd_arg - z_0) / T_z))
                        v_2nd_arg_floor = int(v_2nd_arg)
                        dv_2nd = v_2nd_arg - float(v_2nd_arg_floor)
                        curVal += (t_2nd_high - t_2nd_low) * ((1.0 - dv_2nd)*g[v_2nd_arg_floor, iu] + dv_2nd * g[(v_2nd_arg_floor + 1), iu])
                    else:
                        disc_outer = np.sqrt(disc_outer);
                        # t interval: (b_ti-np.sqrt(disc_outer))*a_ti_inv to (b_ti+np.sqrt(disc_outer))*a_ti_inv
                        # t interval midpoint: b_ti*a_ti_inv

                        # take mid value for interval to find iv
                        v_arg = -(b_ti*a_ti_inv)*tan_beta + z * sec_beta

                        v_arg = max(0.0, min(float(numZ - 1.001), (v_arg - z_0) / T_z))
                        v_arg_floor = int(v_arg)
                        dv = v_arg - float(v_arg_floor)
                        curVal += 2.0*disc_outer*a_ti_inv*((1.0 - dv)*g[v_arg_floor, iu] + dv * g[(v_arg_floor + 1), iu])
            f[k,j] = curVal * np.sqrt(1.0 + tan_beta * tan_beta)


class SymmetricProjectors:
    def __init__(self):
        self.PARALLEL, self.CONE = [0, 1]
        self.reset()
        
    def reset(self):
        self.geometry = None # 0 for parallel, 1 for cone, otherwise undefined
        self.sod = 0.0 # source to object distance, mm
        self.sdd = 0.0 # source to detector distance, mm
        self.numRows = 0 # number of rows in the detector
        self.numCols = 0 # number of columns in the detector
        self.pixelHeight = 0.0 # distance between two detector rows, mm
        self.pixelWidth = 0.0 # distance between two detector columns, mm
        self.centerRow = 0.0 # pixel row index where the central ray hit the detector
        self.centerCol = 0.0 # pixel column index where the ray from the source and through the axis of symmetry hits the detector
        self.axisOfSymmetry = 0.0 # the angular rotation of the axis of symmetry, degrees

        self.tau = 0.0
        
        self.numZ = 0 # number of reconstruction pixels along the axis dimension
        self.numR = 0 # number of reconstruction pixels along the radial dimension
        self.sizeZ = 0.0 # pixel size of reconstruction of the pixel along the axis dimension
        self.sizeR = 0.0 # pixel size of reconstruction of the pixel along the radial dimension
        self.offsetZ = 0.0 # reconstruction image shift in the axis dimension
        self.offsetR = 0.0 # reconstruction image shift in the radial dimension

    def print_params(self):
        if self.geometry == self.PARALLEL:
            print('PARALLEL')
        else:
            print('CONE')
            print('source to object distance: ' + str(self.sod) + ' mm')
            print('source to detector distance: ' + str(self.sdd) + ' mm')
        print('number of detector elements: ' + str(self.numRows) + ' x ' + str(self.numCols))
        print('detector pixel size: ' + str(self.pixelHeight) + ' mm x ' + str(self.pixelWidth) + ' mm')
        print('number of reconstruction pixels: ' + str(self.numZ) + ' x ' + str(self.numR))
        print('reconstruction pixel size: ' + str(self.sizeZ) + ' mm x ' + str(self.sizeR) + ' mm')
        

    def check_params(self):
        if self.geometry != self.PARALLEL and self.geometry != self.CONE:
            return False
        elif self.numRows <= 0 or self.numCols <= 0 or self.pixelHeight <= 0.0 or self.pixelWidth <= 0.0 or self.numZ <= 0 or self.numR <= 0 or self.sizeZ <= 0.0 or self.sizeR <= 0.0:
            return False
        else:
            self.splitSymmetryOnPixelEdge()
            return True
            
    def splitSymmetryOnPixelEdge(self):
        #n = 2*int(np.round(self.r(0)/self.sizeR)) - 1
        n = 2*int(np.round(self.r(0)/self.sizeR-0.5)) + 1
        #print('before: offsetR = ' + str(self.offsetR))
        self.offsetR = 0.5*self.sizeR*(self.numR-1+n)
        #print('after: offsetR = ' + str(self.offsetR))
        
    def allocateProjection(self, val=0.0):
        if self.check_params():
            if val == 0.0:
                return np.zeros((self.numRows, self.numCols), dtype=np.float32)
            else:
                return val*np.ones((self.numRows, self.numCols), dtype=np.float32)
        else:
            return None

    def allocateReconstruction(self, val=0.0):
        if self.check_params():
            if val == 0.0:
                return np.zeros((self.numZ, self.numR), dtype=np.float32)
            else:
                return val*np.ones((self.numZ, self.numR), dtype=np.float32)
        else:
            return None
    
    def row(self, i):
        return (i - self.centerRow)*self.pixelHeight
        
    def col(self, i):
        return (i - self.centerCol)*self.pixelWidth
    
    def r(self, i):
        return (i - 0.5*(self.numR-1))*self.sizeR + self.offsetR

    def z(self, i):
        '''
        if self.geometry == self.CONE:
            return i*self.sizeZ - self.centerRow*(self.sod/self.sdd)*self.pixelHeight + self.offsetZ
        else:
            return i*self.sizeZ - self.centerRow*self.pixelHeight + self.offsetZ
        '''
        if self.geometry == self.CONE:
            z_0 = self.offsetZ - 0.5 * float(self.numZ - 1) * self.sizeZ
        else:
            z_0 = self.offsetZ - self.centerRow * self.pixelHeight
        return i*self.sizeZ + z_0
        
    
    def set_parallelbeam(self, numRows, numCols, pixelHeight, pixelWidth, axisOfSymmetry=0.0, centerRow=None, centerCol=None, numZ=0, numR=0, sizeZ=0.0, sizeR=0.0, offsetZ=0.0, offsetR=0.0):
        self.geometry = self.PARALLEL
        self.numRows = numRows
        self.numCols = numCols
        self.pixelHeight = pixelHeight
        self.pixelWidth = pixelWidth
        if centerRow is None:
            self.centerRow = 0.5*(numRows-1)
        else:
            self.centerRow = centerRow
        if centerCol is None:
            self.centerCol = 0.5*(numCols-1)
        else:
            self.centerCol = centerCol
        self.axisOfSymmetry = axisOfSymmetry
        if numZ <= 0:
            self.numZ = self.numRows
        else:
            self.numZ = numZ
        if numR <= 0:
            self.numR = self.numCols
            #if self.numR % 2 == 1:
            #    self.numR += 1
            #    #print('WARNING: Setting numR to the next largest even number')
        else:
            self.numR = numR
            #if self.numR % 2 == 1:
            #    self.numR += 1
            #    print('WARNING: Setting numR to the next largest even number')
        if sizeZ <= 0.0:
            self.sizeZ = self.pixelHeight
        else:
            self.sizeZ = sizeZ
        if sizeR <= 0.0:
            self.sizeR = self.pixelWidth
        else:
            self.sizeR = sizeR
        self.offsetZ = offsetZ
        self.offsetR = offsetR
        
    
    def set_conebeam(self, numRows, numCols, pixelHeight, pixelWidth, sod, sdd, axisOfSymmetry=0.0, centerRow=None, centerCol=None, numZ=0, numR=0, sizeZ=0.0, sizeR=0.0, offsetZ=None, offsetR=None):
        self.geometry = self.CONE
        self.numRows = numRows
        self.numCols = numCols
        self.pixelHeight = pixelHeight
        self.pixelWidth = pixelWidth
        self.sod = sod
        self.sdd = sdd
        if centerRow is None:
            self.centerRow = 0.5*(numRows-1)
        else:
            self.centerRow = centerRow
        if centerCol is None:
            self.centerCol = 0.5*(numCols-1)
        else:
            self.centerCol = centerCol
        self.axisOfSymmetry = axisOfSymmetry
        if numZ <= 0:
            self.numZ = self.numRows
        else:
            self.numZ = numZ
        
        if numR <= 0:
            self.numR = self.numCols
            #if self.numR % 2 == 1:
            #    self.numR += 1
            #    #print('WARNING: Setting numR to the next largest even number')
        else:
            self.numR = numR
            #if self.numR % 2 == 1:
            #    self.numR += 1
            #    print('WARNING: Setting numR to the next largest even number')
            
        if sizeZ <= 0.0:
            self.sizeZ = self.pixelHeight * self.sod / self.sdd
        else:
            self.sizeZ = sizeZ
        if sizeR <= 0.0:
            self.sizeR = self.pixelWidth * self.sod / self.sdd
        else:
            self.sizeR = sizeR
        if offsetR is None:
            offsetR = 0.0
        if offsetZ is None:
            offsetZ = 0.5 * float(self.numZ - 1) * self.sizeZ - self.centerRow * (self.sod / self.sdd * self.pixelHeight)

        self.offsetZ = offsetZ
        self.offsetR = offsetR
        
    def rampFilter(self, g):
        g_left, g_right = self.splitLeftAndRight(g)
        g_left = self.rampFilter_oneSide(g_left)
        g_right = self.rampFilter_oneSide(g_right)
        g = self.mergeLeftAndRight(g_left, g_right)
        return g
        
    def rampFilter_oneSide(self, g):
    
        from numpy.fft import fft, ifft
        N = int(2.0**np.ceil(np.log2(2 * self.numCols)))
        s_sq = np.array(range(N),dtype=np.float32)-float(N//2)
        s_sq = s_sq**2
        h = 1.0 / (np.pi * (0.25 - s_sq))
        if self.geometry == self.CONE:
            magFactor = self.sod/self.sdd
        else:
            magFactor = 1.0
        h = h / (magFactor*self.pixelWidth)
        H = np.abs(fft(h))
        for n in range(self.numRows):
            q = np.squeeze(g[n,:])
            q = np.real(ifft(fft(q,N)*H))
            g[n,:] = q[0:self.numCols]
        return g
            
    def rayWeight(self, g):
        if self.geometry == self.CONE:
            u = self.col(np.array(range(self.numCols)))/self.sdd
            v = self.row(np.array(range(self.numRows)))/self.sdd
            u,v = np.meshgrid(u,v)
            g = g * (1.0+self.tau/self.sod)/(1.0 + u**2 + v**2)
        return g
        
    def splitLeftAndRight(self, g):
        
        s = self.col(np.array(range(self.numCols)))
        s_conj = s.copy()
        s_conj *= -1.0
        #(i - self.centerCol)*self.pixelWidth
        s_conj_ind = s_conj / self.pixelWidth + self.centerCol
        s_lo = np.array(np.floor(s_conj_ind), dtype=np.int32)
        s_lo[s_lo<0] = 0
        s_lo[s_lo>self.numCols - 1] = self.numCols - 1
        s_hi = s_lo + 1
        s_hi[s_hi>self.numCols - 1] = self.numCols - 1
        s_hi[s_hi<0] = 0
        s_hi = np.array(s_hi, dtype=np.int32)
        ds = s_conj_ind - np.array(s_lo,dtype=np.float32)
        
        g_left = self.allocateProjection()
        g_right = self.allocateProjection()
        
        ind_s_pos = s > 0.0
        ind_s_neg = s < 0.0
        ind_s_zero = s == 0.0
        for irow in range(self.numRows):
            val = g[irow,:]
            val_conj = (1.0 - ds) * val[s_lo] + ds * val[s_hi]
            
            g_right[irow, ind_s_pos] = val[ind_s_pos]
            g_left[irow, ind_s_pos] = val_conj[ind_s_pos]
            
            g_right[irow, ind_s_neg] = val_conj[ind_s_neg]
            g_left[irow, ind_s_neg] = val[ind_s_neg]
            
            g_right[irow, ind_s_zero] = val[ind_s_zero]
            g_left[irow, ind_s_zero] = val[ind_s_zero]
    
        return g_left, g_right
    
    def mergeLeftAndRight(self, g_left, g_right):
        g = self.allocateProjection()
        s = self.col(np.array(range(self.numCols)))
        ind_pos = s >= 0.0
        ind_neg = s < 0.0
        for irow in range(self.numRows):
            g[irow,ind_pos] = g_right[irow,ind_pos]
            g[irow,ind_neg] = g_left[irow,ind_neg]
        return g
        
    def SART(self, g, f, numIter):
        P1 = self.allocateProjection()
        self.project(P1,self.allocateReconstruction(1.0))
        P1[P1<=0.0] = 1.0
        
        Pstar1 = self.allocateReconstruction()
        self.backproject(self.allocateProjection(1.0), Pstar1)
        Pstar1[Pstar1<=0.0] = 1.0
        
        Pd = self.allocateProjection()
        d = self.allocateReconstruction()

        import time
        startTime = time.time()
        for n in range(numIter):
            print('SART iteration ' + str(n+1) + ' of ' + str(numIter))
            self.project(Pd,f)
            Pd = (g-Pd) / P1
            self.backproject(Pd,d)
            f += 0.9*d / Pstar1
            f[f<0.0] = 0.0
        print('average time per iteration: ' + str((time.time()-startTime)/float(numIter)))
        return f
    
    def project(self, g, f):
        if self.check_params() == False:
            return None
        beta = self.axisOfSymmetry
        if self.geometry == self.PARALLEL:
 
            T_v = self.pixelHeight
            v_0 = self.row(0)
            T_u = self.pixelWidth
            u_0 = self.col(0)
            T_z = self.sizeZ
            z_0 = self.z(0)
            T_r = self.sizeR
            r_0 = self.r(0)
        
            project_parallel(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0)
            return g
        elif self.geometry == self.CONE:
            
            T_v = self.pixelHeight / self.sdd
            v_0 = self.row(0) / self.sdd
            T_u = self.pixelWidth / self.sdd
            u_0 = self.col(0) / self.sdd
            T_r = self.sizeR
            r_0 = self.r(0)
            T_z = self.sizeZ
            z_0 = self.z(0)
            sod = self.sod
            tau = self.tau
        
            project_cone(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0, sod, tau)
            return g
        else:
            return None
    
    def backproject(self, g, f):
        if self.check_params() == False:
            return None
        beta = self.axisOfSymmetry
        if self.geometry == self.PARALLEL:
        
            T_v = self.pixelHeight
            v_0 = self.row(0)
            T_u = self.pixelWidth
            u_0 = self.col(0)
            T_z = self.sizeZ
            z_0 = self.z(0)
            T_r = self.sizeR
            r_0 = self.r(0)
            
            backproject_parallel(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0)
            return f
        elif self.geometry == self.CONE:
        
            T_v = self.pixelHeight / self.sdd
            v_0 = self.row(0) / self.sdd
            T_u = self.pixelWidth / self.sdd
            u_0 = self.col(0) / self.sdd
            T_r = self.sizeR
            r_0 = self.r(0)
            T_z = self.sizeZ
            z_0 = self.z(0)
            sod = self.sod
            tau = self.tau
        
            backproject_cone(g, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0, sod, tau)
            return f
        else:
            return None
            
    def FBP(self, g, f):
        if self.check_params() == False:
            return None
        q = g.copy()
        q = self.rayWeight(q)
        q = self.rampFilter(q)
        beta = self.axisOfSymmetry
        if self.geometry == self.PARALLEL:
        
            T_v = self.pixelHeight
            v_0 = self.row(0)
            T_u = self.pixelWidth
            u_0 = self.col(0)
            T_z = self.sizeZ
            z_0 = self.z(0)
            T_r = self.sizeR
            r_0 = self.r(0)
            
            inverse_transform(q, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0, 0.0, 0.0)
            return f
        elif self.geometry == self.CONE:
        
            T_v = self.pixelHeight / self.sdd
            v_0 = self.row(0) / self.sdd
            T_u = self.pixelWidth / self.sdd
            u_0 = self.col(0) / self.sdd
            T_r = self.sizeR
            r_0 = self.r(0)
            T_z = self.sizeZ
            z_0 = self.z(0)
            sod = self.sod
            tau = self.tau
        
            inverse_transform(q, f, beta, T_v, v_0, T_u, u_0, T_z, z_0, T_r, r_0, sod, tau)
            return f
        else:
            return None


''' Example Usage
import matplotlib.pyplot as plt

# Make an instance of the SymmetricProjectors class
flash = SymmetricProjectors()

# Set the geometry parameters
# There are other parameters in these functions which are currently just being set to their default values
# You can also set the reconstruction pixel size,
# specify the position of the detector (shifting it up/down or left/right), and more
# See the SymmetricProjectors constructor for a description of parameters
numCols = 512 # number of detector columns
pixelSize = 512/numCols # detector pixel size, mm
numRows = numCols # number of detector rows
tiltAngle = 10.0 # tilt angle, degrees
sod = 1100.0 # source to object distance, mm (only needed for cone-beam)
sdd = 1400.0 # source to detector distance, mm (only needed for cone-beam)
#flash.set_parallelbeam(numRows, numCols, pixelSize, pixelSize, tiltAngle) # parallel-beam
flash.set_conebeam(numRows, numCols, pixelSize, pixelSize, sod, sdd, tiltAngle) # cone-beam

# Allocate numpy arrays for the projection data (g) and reconstruction image (f)
# These functions are just for convenience and you don't have to use them
g = flash.allocateProjection()
f = flash.allocateReconstruction()

# For testing purposes, we will set f to a sphere and forward project it
r = np.array(range(flash.numR),dtype=np.float32)
z = np.array(range(flash.numZ),dtype=np.float32)
for i in range(flash.numR):
    r[i] = flash.r(i)
for i in range(flash.numZ):
    z[i] = flash.z(i)
r,z = np.meshgrid(r,z)
f[r**2+z**2 <= (100.0)**2] = 1.0

flash.project(g,f)

# Now let's try to reconstruct this sphere with SART
f[:] = 0.0 # initial to the reconstruction to zero, so we are not cheating
flash.SART(g,f,100)
#flash.FBP(g,f)

# Display the result
plt.imshow(f)
plt.show()
#'''
