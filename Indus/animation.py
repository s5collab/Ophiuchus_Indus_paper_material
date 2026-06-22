import os
os.chdir("/Users/yyan0723/Desktop/S5/indus/")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter 
import astropy.units as u
from astropy.table import Table
import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')
import gala.coordinates as gc
import agama
agama.setUnits(length=1, mass=1, velocity=1)  # 1 kpc, 1 Msun, 1 km/s
tunit = 977.79222168/1000 #Gyr
dt = 1/(tunit*1000) # 1 Myr

points = coord.SkyCoord(ra=[ 340, 80, ]*u.deg, dec=[ -60, -60, ]*u.deg)
frame = gc.GreatCircleICRSFrame.from_endpoints(points[0], points[1])

# potential
potMW      = agama.Potential('/Users/yyan0723/Desktop/S5/bar/MWPotentialHunter24_axi.ini')
massLMC    = 1.5e11
radiusLMC  = (massLMC/1e11)**0.6 * 8.5
potLMC     = agama.Potential(
    type              = 'spheroid',
    mass              = massLMC,
    scaleradius       = radiusLMC,
    outercutoffradius = radiusLMC*10,
    gamma             = 1,
    beta              = 3)

potacc  = agama.Potential(type='UniformAcceleration', file='accel.txt')
potLMCm = agama.Potential(potential=potLMC, center='trajlmc.txt')  # potential of the moving LMC
# finally, the total time-dependent potential in the non-inertial MW-centric reference frame
potMWLMC= agama.Potential(potMW, potLMCm, potacc)

# indus orbit
old_orbit = Table.read('old_indus_orbit.fits')[5000]
prog_w0 = np.array([ old_orbit['x'],old_orbit['y'],old_orbit['z'],old_orbit['vx'],old_orbit['vy'],old_orbit['vz'], ])

past_time, past_orbit = agama.orbit(ic= prog_w0, potential= potMWLMC, time= -5/tunit, timestart= 0, trajsize= 5000+1, )
past_time = past_time[::-1]
past_orbit = past_orbit[::-1]   

future_time, future_orbit = agama.orbit(ic= prog_w0, potential= potMWLMC, time= 100*dt, timestart= 0, trajsize= 100+1, )
prog_time, prog_orbit = np.hstack((past_time[:-1],future_time)), np.vstack((past_orbit[:-1],future_orbit))

prog_Lz = prog_orbit[:,0]*prog_orbit[:,4]-prog_orbit[:,1]*prog_orbit[:,3]
prog_E = potMWLMC.potential(prog_orbit[:,:3], t= prog_time) + 0.5 * np.sum(prog_orbit[:,3:]**2,axis=1)

# snapshot
combs = [ [35666,'cored'],  [55670,'cuspy'],  [47103,'stars'], ]
idx1, typ =  combs[2]

ts = np.load('./nbody/times_{}.npy'.format(typ))
xv = np.load('./nbody/xvs_{}.npy'.format(typ))
part_Lz, part_E = [], []
for i in range(len(ts)):
    part_Lz.append( xv[i][:,0]*xv[i][:,4]-xv[i][:,1]*xv[i][:,3] )
    part_E.append( potMWLMC.potential(xv[i][:,:3], t= ts[i]) + 0.5 * np.sum(xv[i][:,3:]**2, axis=1) )
# interpolate snapshot time
prog_xvLzE = np.column_stack([ np.interp(ts, prog_time, prog_orbit[:,0]),
                                np.interp(ts, prog_time, prog_orbit[:,1]),
                                np.interp(ts, prog_time, prog_orbit[:,2]),
                                np.interp(ts, prog_time, prog_orbit[:,3]),
                                np.interp(ts, prog_time, prog_orbit[:,4]),
                                np.interp(ts, prog_time, prog_orbit[:,5]),
                                np.interp(ts, prog_time, prog_Lz),
                                np.interp(ts, prog_time, prog_E),])
# orbital plane, new orientation
LL = np.cross(prog_xvLzE[:,0:3],prog_xvLzE[:,3:6])
r_gc = np.linalg.norm(prog_xvLzE[:,0:3], axis=-1)
z_unit = LL / np.linalg.norm(LL, axis=-1).reshape(-1,1)
x_unit = prog_xvLzE[:,0:3] / r_gc.reshape(-1,1)
y_unit = np.cross(z_unit,x_unit)
part_newxyz = []
orbit_newxyz = []
for i in range(len(ts)):
    partx_ = np.sum(xv[i][:,0:3]*x_unit[i],axis=1)
    party_ = np.sum(xv[i][:,0:3]*y_unit[i],axis=1)
    partz_ = np.sum(xv[i][:,0:3]*z_unit[i],axis=1)
    part_newxyz.append( np.column_stack(( partx_-r_gc[i], party_, partz_, )) )

    orbit_ = prog_orbit[ np.abs(prog_time-ts[i])*tunit < 100/1e3 ]
    orbitx_ = np.sum(orbit_[:,0:3]*x_unit[i],axis=1)
    orbity_ = np.sum(orbit_[:,0:3]*y_unit[i],axis=1)
    orbitz_ = np.sum(orbit_[:,0:3]*z_unit[i],axis=1)
    orbit_newxyz.append( np.column_stack(( orbitx_-r_gc[i], orbity_, orbitz_, )) )

# LMC
trajlmc = np.loadtxt('trajlmc.txt')
lmc_rgc = np.interp(ts, trajlmc[:,0], np.linalg.norm(trajlmc[:,1:4], axis=-1), )

# --------------animate-----------------
fig,axs = plt.subplots(1,3, figsize=[17,10],)
#plt.subplots_adjust(wspace=0.07)
axs = axs.ravel()

xy_orbit = axs[0].plot([], [], ls='-', lw=0.3, color='b', )[0]
xy_part = axs[0].plot([], [], ls='', marker='.', ms=0.1, color='k', alpha=0.1, )[0]
xy_part1 = axs[0].plot([], [], ls='', marker='*', ms=5, color='g', )[0]
axs[0].plot(0, 0, ls='', marker='o', ms=5, color='r', )

zy_orbit = axs[1].plot([], [], ls='-', lw=0.3, color='b', )[0]
zy_part = axs[1].plot([], [], ls='', marker='.', ms=0.1, color='k', alpha=0.1, )[0]
zy_part1 = axs[1].plot([], [], ls='', marker='*', ms=5, color='g', )[0]
axs[1].plot(0, 0, ls='', marker='o', ms=5, color='r', )

LzE_part = axs[2].plot([], [], ls='', marker='.', ms=0.1, color='k', alpha=0.1, )[0]
LzE_part1 = axs[2].plot([], [], ls='', marker='*', ms=5, color='g', )[0]
LzE_prog = axs[2].plot([], [], ls='', marker='o', ms=5, color='r', )[0]

axs[0].set_xlabel('X (kpc)');axs[0].set_ylabel('Y (kpc)');axs[0].set_xlim(-10,10);axs[0].set_ylim(-20,20)
axs[1].set_xlabel('Z (kpc)');axs[1].set_ylabel('Y (kpc)');axs[1].set_xlim(-10,10);axs[1].set_ylim(-20,20)
axs[2].set_xlabel(r'L$_z$ ($\times 10^3$ kpc km s$^{-1}$)');axs[2].set_ylabel(r'Energy ($\times 10^5$ km$^2$ s$^{-2}$)');axs[2].yaxis.set_label_position("right");axs[2].yaxis.tick_right()

def update(frame):
    xy_orbit.set_data(orbit_newxyz[frame][:,0],orbit_newxyz[frame][:,1])
    xy_part.set_data(part_newxyz[frame][:,0],part_newxyz[frame][:,1])
    xy_part1.set_data([part_newxyz[frame][idx1,0]],[part_newxyz[frame][idx1,1]])

    zy_orbit.set_data(orbit_newxyz[frame][:,2],orbit_newxyz[frame][:,1])
    zy_part.set_data(part_newxyz[frame][:,2],part_newxyz[frame][:,1])
    zy_part1.set_data([part_newxyz[frame][idx1,2]],[part_newxyz[frame][idx1,1]])

    LzE_part.set_data(part_Lz[frame]/1e3,part_E[frame]/1e5)
    LzE_part1.set_data([part_Lz[frame][idx1]/1e3],[part_E[frame][idx1]/1e5])
    LzE_prog.set_data([prog_xvLzE[frame,-2]/1e3],[prog_xvLzE[frame,-1]/1e5])
    axs[2].set_xlim((prog_xvLzE[frame,-2]-0.5e3)/1e3, (prog_xvLzE[frame,-2]+0.5e3)/1e3)
    axs[2].set_ylim((prog_xvLzE[frame,-1]-1e4)/1e5, (prog_xvLzE[frame,-1]+1e4)/1e5)

    axs[0].set_title(rf'time = { round(ts[frame]*tunit*1000) } Myr')
    axs[1].set_title(rf'r$_\mathrm{{gc,Indus}}$ = { round( r_gc[frame], 2) } kpc')
    axs[2].set_title(rf'r$_\mathrm{{gc,LMC}}$ = { round( lmc_rgc[frame], 2) } kpc')

    return (xy_orbit,xy_part,xy_part1,zy_orbit,zy_part,zy_part1,LzE_part,LzE_part1,LzE_prog,)

# --- Save as MP4 using ffmpeg ---
ani = FuncAnimation(fig=fig, func=update, frames=len(ts), interval=150, repeat=False, blit=True, )
#plt.show()
writer = FFMpegWriter(
    fps=30,
    bitrate=2000,                 # adjust if file size/quality needs tuning
    codec='libx264',              # H.264
    extra_args=['-pix_fmt', 'yuv420p']  # ensures compatibility (PowerPoint/QuickTime)
)

ani.save('gadget4_nbody_dissolution_{}.mp4'.format(typ), writer=writer, dpi=150)
print("Saved to mp4 for {}".format(typ))
