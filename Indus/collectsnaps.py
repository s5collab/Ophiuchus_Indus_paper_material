
typ = 'stars'
import os
os.chdir( './{}/'.format(typ) )
import numpy as np
import h5py as h5
import glob

snaplist = sorted( glob.glob("snapshot_*.hdf5") )
times = []
xvs = []
for i in range(len(snaplist)):
    with h5.File(snaplist[i],'r') as f:
        times.append( f['Header'].attrs["Time"] ) # in 0.98Gyr
        sort_ids = np.argsort( f['PartType3/ParticleIDs'][:] )
        xvs.append( np.column_stack(( f['PartType3/Coordinates'][:], f['PartType3/Velocities'][:], ))[sort_ids] )

np.save('times_{}.npy'.format(typ), np.array(times))
np.save('xvs_{}.npy'.format(typ), np.array(xvs))
print('{} done'.format(typ))
