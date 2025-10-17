#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control methods.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
from python_vehicle_simulator.lib.guidance import refModel3, refModel2
from python_vehicle_simulator.lib.gnc import ssa, Rzyx

# SISO PID pole placement
def PIDpolePlacement(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    m,
    d,
    k,
    wn_d,
    zeta_d,
    wn,
    zeta,
    r,
    v_max,
    sampleTime,
):

    # PID gains based on pole placement
    Kp = m * wn ** 2.0 - k
    Kd = m * 2.0 * zeta * wn - d
    Ki = (wn / 10.0) * Kp

    # PID control law
    u = -Kp * e_x - Kd * e_v - Ki * e_int

    # Integral error, Euler's method
    e_int += sampleTime * e_x

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime)

    return u, e_int, x_d, v_d, a_d


# MIMO nonlinear PID pole placement
def DPpolePlacement(
    e_int, M3, D3, eta3, nu3, x_d, y_d, psi_d, wn, zeta, eta_ref, sampleTime
):

    # PID gains based on pole placement
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))
    
    Kp = wn @ wn @ M3_diag
    Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
    Ki = (1.0 / 10.0) * wn @ Kp

    # DP control law - setpoint regulation
    e = eta3 - np.array([x_d, y_d, psi_d])
    e[2] = ssa(e[2])
    R = Rzyx(0.0, 0.0, eta3[2])
    tau = (
        - np.matmul((R.T @ Kp), e)
        - np.matmul(Kd, nu3)
        - np.matmul((R.T @ Ki), e_int)
    )

    # Low-pass filters, Euler's method
    T = 5.0 * np.array([1 / wn[0][0], 1 / wn[1][1], 1 / wn[2][2]])
    x_d += sampleTime * (eta_ref[0] - x_d) / T[0]
    y_d += sampleTime * (eta_ref[1] - y_d) / T[1]
    psi_d += sampleTime * (eta_ref[2] - psi_d) / T[2]

    # Integral error, Euler's method
    e_int += sampleTime * e

    return tau, e_int, x_d, y_d, psi_d

# Heading autopilot - Intergral SMC (Equation 16.479 in Fossen 2021)
def integralSMC(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    T_nomoto,
    K_nomoto,
    wn_d,
    zeta_d,
    K_d,
    K_sigma,
    lam,
    phi_b,
    r,
    v_max,
    sampleTime,
):

    # Sliding surface
    v_r_dot = a_d - 2 * lam * e_v - lam ** 2 * ssa(e_x)
    v_r     = v_d - 2 * lam * ssa(e_x) - lam ** 2 * e_int
    sigma   = e_v + 2 * lam * ssa(e_x) + lam ** 2 * e_int

    #  Control law
    if abs(sigma / phi_b) > 1.0:
        delta = ( T_nomoto * v_r_dot + v_r - K_d * sigma 
                 - K_sigma * np.sign(sigma) ) / K_nomoto
    else:
        delta = ( T_nomoto * v_r_dot + v_r - K_d * sigma 
                 - K_sigma * (sigma / phi_b) ) / K_nomoto

    # Integral error, Euler's method
    e_int += sampleTime * ssa(e_x)

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime)

    return delta, e_int, x_d, v_d, a_d


# Velocity controller with 2nd-order reference model and pole placement
def VelocityPolePlacement(
    u,
    r,
    u_d,
    u_dot_d,
    r_d,
    r_dot_d,
    u_int,
    r_int,
    m_u,
    d_u,
    m_r,
    d_r,
    wn_d_u,
    zeta_d_u,
    wn_d_r,
    zeta_d_r,
    wn_u,
    zeta_u,
    wn_r,
    zeta_r,
    u_ref,
    r_ref,
    u_max,
    r_max,
    sampleTime,
):
    """
    Velocity controller for surge (u) and yaw rate (r) using pole placement.
    
    Control law (Feedforward + PI feedback):
        tau_X = m_u * u_dot_d + d_u * u_d - Kp_u * (u - u_d) - Ki_u * integral(u - u_d)
        tau_N = m_r * r_dot_d + d_r * r_d - Kp_r * (r - r_d) - Ki_r * integral(r - r_d)
    
    Note: Integral term automatically compensates for nonlinear damping and model uncertainties.
    The system's natural damping (d_u, d_r) provides sufficient damping.
    
    Inputs:
        u, r: current surge velocity and yaw rate
        u_d, u_dot_d: desired surge velocity and acceleration states
        r_d, r_dot_d: desired yaw rate and angular acceleration states
        u_int, r_int: integral error states
        m_u, d_u: surge mass and damping
        m_r, d_r: yaw inertia and damping
        wn_d_u, zeta_d_u: reference model parameters for surge
        wn_d_r, zeta_d_r: reference model parameters for yaw rate
        wn_u, zeta_u: controller parameters for surge (pole placement)
        wn_r, zeta_r: controller parameters for yaw rate (pole placement)
        u_ref, r_ref: velocity references (setpoints)
        u_max, r_max: velocity saturation limits
        sampleTime: sampling time
    
    Returns:
        tau_X: surge force
        tau_N: yaw moment
        u_d, u_dot_d: updated desired surge velocity and acceleration
        r_d, r_dot_d: updated desired yaw rate and angular acceleration
        u_int, r_int: updated integral states
    """
    
    # 2nd-order reference model for surge velocity
    [u_d, u_dot_d] = refModel2(u_d, u_dot_d, u_ref, wn_d_u, zeta_d_u, u_max, sampleTime)
    
    # 2nd-order reference model for yaw rate
    [r_d, r_dot_d] = refModel2(r_d, r_dot_d, r_ref, wn_d_r, zeta_d_r, r_max, sampleTime)
    
    # Tracking errors
    e_u = u - u_d
    e_r = r - r_d
    
    # Integral states (with anti-windup)
    u_int += sampleTime * e_u
    r_int += sampleTime * e_r
    
    # Anti-windup saturation
    u_int_max = 100.0  # reasonable limit
    r_int_max = 50.0
    u_int = max(min(u_int, u_int_max), -u_int_max)
    r_int = max(min(r_int, r_int_max), -r_int_max)
    
    # PI gains based on pole placement
    # For first-order system with PI control:
    # Desired poles: s = -wn (proportional) and s = -wn/5 (integral)
    
    # Surge controller
    Kp_u = m_u * wn_u - d_u
    Ki_u = Kp_u * wn_u / 5.0  # integral gain
    
    # Yaw rate controller  
    Kp_r = m_r * wn_r - d_r
    Ki_r = Kp_r * wn_r / 5.0  # integral gain
    
    # Control law: Feedforward + PI feedback
    tau_X = m_u * u_dot_d + d_u * u_d - Kp_u * e_u - Ki_u * u_int
    tau_N = m_r * r_dot_d + d_r * r_d - Kp_r * e_r - Ki_r * r_int
    
    return tau_X, tau_N, u_d, u_dot_d, r_d, r_dot_d, u_int, r_int

