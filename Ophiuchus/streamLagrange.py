# -*- coding: utf-8 -*-
"""
author: yong yang

time: 18:55 8/8/2024
"""

import numpy as np, agama

def mockstreamLagrange(prog_w0, prog_mass0, prog_pot, v_disp, otherhalo_pot, MW_pot, n_particles, time_start, dt, n_steps, random_state, ):
    # random_state = numpy.random.RandomState(seed=None)
    # unit of time is 977.79222168/1000 Gyr, so 1/977.79222168/1000 is 1Gyr
    # first is to integrate progenitor orbit
    if otherhalo_pot is None:
        pot_total = MW_pot
    else:
        pot_total = agama.Potential(otherhalo_pot, MW_pot)
    prog_timestamp, prog_orbit = agama.orbit(ic= prog_w0, 
            potential= pot_total, time= dt*n_steps, timestart= time_start, trajsize= 1+n_steps)
    # if time is from now back to past, reverse it
    if prog_timestamp[1] < prog_timestamp[0]:
        prog_orbit = prog_orbit[::-1]
        prog_timestamp = prog_timestamp[::-1]
    # Jacobi radius
    prog_mass = np.interp(prog_timestamp, [prog_timestamp.min(),prog_timestamp.max()], [prog_mass0,0], )
    x, y, z, vx, vy, vz = prog_orbit.T
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x*x + y*y + z*z)**0.5
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    der = pot_total.forceDeriv(prog_orbit[:,0:3], t= prog_timestamp)[1]
    d2Phi_dr2 = -(x**2  * der[:,0] + y**2  * der[:,1] + z**2  * der[:,2] +
                  2*x*y * der[:,3] + 2*y*z * der[:,4] + 2*z*x * der[:,5]) / r**2
    Omega = L / r**2
    r_j = (agama.G * prog_mass / (Omega**2 - d2Phi_dr2))**(1./3)
    # sample at each timestampe
    total_particles = 2*n_particles*n_steps
    particles_w0 = np.zeros( (total_particles,6) )
    particles_timestart = np.zeros(total_particles)
    particles_time = np.zeros(total_particles)
    particles_type = np.full(total_particles, True)
    j = 0  # batch number
    for i in range(n_steps):
        # L2 point
        prog_w1 = agama.getCelestialCoords(*prog_orbit[i][:3])
        n_particles_w0 = np.zeros( (n_particles,6) )
        n_particles_w0[:,:3] = np.array( agama.getCartesianCoords(prog_w1[0],prog_w1[1],prog_w1[2] + r_j[i]) )
        n_particles_w0[:,3:] = prog_orbit[i][3:] + random_state.normal(loc=0., scale= v_disp, size= (n_particles,3) )
        particles_w0[j*n_particles:(j+1)*n_particles] = n_particles_w0
        particles_timestart[j*n_particles:(j+1)*n_particles] = prog_timestamp[i]
        particles_time[j*n_particles:(j+1)*n_particles] = np.abs(dt)*(n_steps-i)
        E_total = pot_total.potential(n_particles_w0[:,:3], t= prog_timestamp[i]) + 0.5 * np.sum(n_particles_w0[:,3:]**2, axis=1)
        E_0 = pot_total.potential(prog_orbit[i][:3], t= prog_timestamp[i]) + 0.5 * np.sum(prog_orbit[i][3:]**2)
        particles_type[j*n_particles:(j+1)*n_particles] = E_total > E_0
        j = j+1
        # L3 point
        n_particles_w0 = np.zeros( (n_particles,6) )
        n_particles_w0[:,:3] = np.array( agama.getCartesianCoords(prog_w1[0],prog_w1[1],prog_w1[2] - r_j[i]) )
        n_particles_w0[:,3:] = prog_orbit[i][3:] + random_state.normal(loc=0., scale= v_disp, size= (n_particles,3) )
        particles_w0[j*n_particles:(j+1)*n_particles] = n_particles_w0
        particles_timestart[j*n_particles:(j+1)*n_particles] = prog_timestamp[i]
        particles_time[j*n_particles:(j+1)*n_particles] = np.abs(dt)*(n_steps-i)
        E_total = pot_total.potential(n_particles_w0[:,:3], t= prog_timestamp[i]) + 0.5 * np.sum(n_particles_w0[:,3:]**2, axis=1)
        particles_type[j*n_particles:(j+1)*n_particles] = E_total > E_0
        j = j+1
    # integrate particles
    scale = np.column_stack((prog_timestamp, np.interp(prog_timestamp,[prog_timestamp.min(),prog_timestamp.max()],[1,0]), np.ones(len(prog_timestamp)),))
    center = np.column_stack((prog_timestamp, prog_orbit,))
    stream_particles = agama.orbit(ic= particles_w0, 
            potential= agama.Potential(pot_total, agama.Potential(potential=prog_pot, scale=scale, center=center,)), 
            time= particles_time, timestart= particles_timestart, trajsize=1, verbose=False, )   # points with integrating time = 0 will be removed, i.e., blank array
    # return particles posvel
    return np.vstack(stream_particles[:,1]), particles_timestart*977.79222168/1000, np.array([ 'trailing' if boolx else 'leading' for boolx in particles_type ])

print("ready to stream it")