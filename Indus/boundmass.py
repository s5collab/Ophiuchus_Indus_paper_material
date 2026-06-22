# to compute bound mass for N-body sims
# author: yy

import os
typ = 'cored'
os.chdir(f'./{typ}/')
import numpy as np
import h5py as h5
from pytreegrav import Potential as treepot


def find_bound_particles(particles, CoM=None, n_iter=10, ):
    """
    Identifies bound particles using iterative unbinding. Assumes units: kpc, km/s, 1 M_sun (standard gadget units)
    """
    parts_bound = particles.copy()
    # Boolean mask of particles we think are bound (starts with all True)
    is_bound_old = np.ones(len(parts_bound), dtype=bool)
    print(f"Starting with {np.sum(is_bound_old)} particles")

    # converge times judge
    n_converge = 0

    for i in range(n_iter):
        # whether fully disrupt
        if len(parts_bound) < 10:
            print("Dwarf has fully disrupted!")
            return parts_bound
        # derive CoM using weighted pos&vel, OR directly use prog_orbit
        if CoM is None:
            com_pos = np.average(parts_bound[:,0:3], axis=0, weights= parts_bound[:,-2], )
            com_vel = np.average(parts_bound[:,3:6], axis=0, weights= parts_bound[:,-2], )
        else:
            com_pos, com_vel = CoM[0:3], CoM[3:6]
        # --- B. Calculate Kinetic Energy (Relative) ---
        k_energy = 0.5 * np.sum((parts_bound[:,3:6] - com_vel)**2, axis=1)
        # --- C. Calculate Potential Energy (Self-Gravity) ---        
        # Note: If N is large (>10k), use treegrav. If small, use brute force.
        phi = treepot( parts_bound[:,0:3]-com_pos, parts_bound[:,-2], softening=parts_bound[:,-1], G=4.3009e-6, )
        # --- D. Total Energy & Cut ---
        total_energy = k_energy + phi
        # Identify who is bound NOW and update particles
        is_bound_new = total_energy < 0
        parts_bound = parts_bound[is_bound_new]
        # judge if converge
        n_bound_old = np.sum(is_bound_old)
        n_bound_new = np.sum(is_bound_new)
        print(f"Iteration {i}: {n_bound_new} bound particles found.")

        if abs(n_bound_old-n_bound_new) < 10 :
            n_converge = n_converge + 1

        if n_converge > 2:
            print("Converged.")
            break

        is_bound_old = is_bound_new

    return parts_bound


prog_orbits = np.load('/suphys/yyan0723/silo_yyan0723/agama_nbody/auxiliary/prog_orbits.npy')
snap_base = "snapshot_"
snap_nums = range(0,495,)
bound_mass_star, bound_mass_halo = [],[]
times = []

for n in snap_nums:
    filename = f"{snap_base}{n:03d}.hdf5"

    with h5.File(filename,'r') as f:
        time = f['Header'].attrs["Time"] # in 0.98Gyr
        sort_id1 = np.argsort( f['PartType1/ParticleIDs'][:] )
        part1 = np.column_stack(( f['PartType1/Coordinates'][:], f['PartType1/Velocities'][:], f['PartType1/ParticleIDs'][:], 
                                  f['PartType1/Masses'][:], np.full(len(sort_id1), 0.1,), ))[sort_id1]

        sort_id3 = np.argsort( f['PartType3/ParticleIDs'][:] )
        part3 = np.column_stack(( f['PartType3/Coordinates'][:], f['PartType3/Velocities'][:], f['PartType3/ParticleIDs'][:], 
                                  f['PartType3/Masses'][:], np.full(len(sort_id3), 0.005,), ))[sort_id3]
    f.close()
    # pos, vel, id, mass, softening
    parts = np.vstack((part1, part3,))

    prog_orbit = np.array([ np.interp(time, prog_orbits[:,0], prog_orbits[:,i]) for i in range(1,7) ])

    dX_from_orbit = np.linalg.norm(parts[:,0:3] - prog_orbit[0:3], axis=1)
    dV_from_orbit = np.linalg.norm(parts[:,3:6] - prog_orbit[3:6], axis=1)
    # search radius decline over time
    Rmax = np.interp(time, [-5,0], [10,1],)
    Vmax = 500
    parts = parts[ (dX_from_orbit<Rmax) & (dV_from_orbit<Vmax) ]

    parts_bound = find_bound_particles(parts, CoM=prog_orbit, n_iter=1, )

    idx_stellar = parts_bound[:,-1] < 0.01
    if np.sum(idx_stellar) > 0:
        bound_mass_star.append( parts_bound[:,-2][idx_stellar].sum() )
    else:
        bound_mass_star.append(0)

    idx_halo = parts_bound[:,-1] > 0.01
    if np.sum(idx_halo) > 0:
        bound_mass_halo.append( parts_bound[:,-2][idx_halo].sum() )
    else:
        bound_mass_halo.append(0)

    times.append(time)
    
    print(f'{filename} done')

np.save(f'mass_time_{typ}.npy', np.column_stack(( times, bound_mass_star, bound_mass_halo, )))

print('ALL DONE.')

