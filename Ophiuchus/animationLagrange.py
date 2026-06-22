import os
os.chdir("/Users/yyan0723/Desktop/S5/bar/")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')
import agama
agama.setUnits(length=1, mass=1, velocity=1)  # 1 kpc, 1 Msun, 1 km/s

tunit = 977.79222168/1000 #Gyr
c1 = float(tunit*u.Gyr*u.km/(u.s*u.kpc))
c2 = float(0.5*u.km/(u.s*u.kpc*u.Gyr)*(tunit*u.Gyr)**2)
def makeRotation(phi0, omega0, alpha, timearray):
    # phi0: deg, omega0: km/s/kpc, alpha: km/s/kpc/Gyr
    phi1 = phi0*np.pi/180 + c1*omega0*timearray + c2*alpha*timearray**2
    return np.column_stack((timearray,phi1))
# make barred potential
bar = agama.Potential("MWPotentialHunter24_full.ini")
phi0, omega0, alpha = -30, -35, 5.5 # omega0<0 in km/s/kpc
timestamps = np.arange(-3, 0+0.001, 0.001)/tunit
rotation = makeRotation(phi0, omega0, alpha, timestamps)
pot_bar = agama.Potential(potential=bar, rotation=rotation)
pot_axi = agama.Potential("MWPotentialHunter24_axi.ini")
MW_pot = pot_axi
#MW_pot = pot_bar

with fits.open("ophiuchus_members_revised.fits") as file:
    ophiu = file[1].data
cluster = coord.SkyCoord(ra= np.median(ophiu['ra']) *u.degree, 
                        dec=  np.median(ophiu['dec']) *u.degree,
                        distance= 7.9 *u.kpc,
                        pm_ra_cosdec= np.median(ophiu['pmra']) *u.mas/u.yr,
                        pm_dec= np.median(ophiu['pmdec']) *u.mas/u.yr,
                        radial_velocity= np.median(ophiu['rv']) *u.km/u.s, frame="icrs").galactocentric
prog_w0 = np.array([cluster.x.to_value(),cluster.y.to_value(),cluster.z.to_value(),
        cluster.v_x.to_value(),cluster.v_y.to_value(),cluster.v_z.to_value(),])
prog_mass0 = 2e4
prog_pot  = agama.Potential(type='Plummer', mass=prog_mass0, scaleRadius=30/1000)
time_start= 0
dt = - 1/(tunit*1000)  #1 Myr
n_steps = 3000
# prog orbit
prog_timestamp, prog_orbit = agama.orbit(ic= prog_w0, potential= MW_pot, time= dt*n_steps, timestart= 0, trajsize= 1+n_steps)
prog_timestamp, prog_orbit = prog_timestamp[::-1], prog_orbit[::-1]

with fits.open('one_group_axi.fits') as file:
    groups = file[1].data
# all particles divided into 3 groups by release time: -3 to -2.5 (old_group), -2.5 to -1.5 (mid_group), and -1.5 to 0 (young_group) Gyr.
idx_old = groups['release_time']<=-2.5+0.5/1e3
idx_mid = (groups['release_time']>-2.5-0.5/1e3)&(groups['release_time']<=-1.5+0.5/1e3)
idx_young = groups['release_time']>-1.5-0.5/1e3

idx_lead = groups['LorT']=='leading'
idx_trail = groups['LorT']=='trailing'
#%%
lead = groups[  idx_lead]
trail = groups[  idx_trail]
#%%
axs = plt.subplots(2,3, figsize=[14,9],)[1].ravel()
plt.ion()
for i in range(len(prog_timestamp)):
    for ax in axs:
        ax.cla()
    axs[0].scatter(lead['x'][:,i], lead['y'][:,i], marker='o', s=1, linewidths=0, edgecolors='none', c= 'r')
    axs[0].scatter(trail['x'][:,i], trail['y'][:,i], marker='o', s=1, linewidths=0, edgecolors='none', c= 'b')
    axs[0].plot(prog_orbit[i,0], prog_orbit[i,1], marker='*',markersize=5,color='k')
    axs[0].set_xlim(prog_orbit[i,0]-3, prog_orbit[i,0]+3)
    axs[0].set_ylim(prog_orbit[i,1]-3, prog_orbit[i,1]+3)
    axs[0].set_xlabel('x (kpc)')
    axs[0].set_ylabel('y (kpc)')
    axs[0].text(prog_orbit[i,0]-3+0.5, prog_orbit[i,1]+3-0.5, "{} Gyr".format(np.round(prog_timestamp[i]*tunit,2)))
    #axs[0].add_patch(Ellipse(xy=(0,0),width=4,height=2,angle=np.rad2deg(np.interp(prog_timestamp[i], rotation[:,0], rotation[:,1])),zorder=-1,),)
    axs[0].add_patch(Ellipse(xy=(0,0),width=2,height=2))

    axs[1].scatter(lead['x'][:,i], lead['z'][:,i], marker='o', s=1, linewidths=0, edgecolors='none', c= 'r')
    axs[1].scatter(trail['x'][:,i], trail['z'][:,i], marker='o', s=1, linewidths=0, edgecolors='none', c= 'b')
    axs[1].plot(prog_orbit[i,0], prog_orbit[i,2], marker='*',markersize=5,color='k')
    axs[1].set_xlim(prog_orbit[i,0]-3, prog_orbit[i,0]+3)
    axs[1].set_ylim(prog_orbit[i,2]-3, prog_orbit[i,2]+3)
    axs[1].set_xlabel('x (kpc)')
    axs[1].set_ylabel('z (kpc)')
    axs[1].text(prog_orbit[i,0]-3+0.5, prog_orbit[i,2]+3-0.5, "r_gc= {} kpc".format(np.round(np.sum(prog_orbit[i,0:3]**2)**0.5, 2) ) )
    axs[1].add_patch(Ellipse(xy=(0,0),width=2,height=2))

    axs[2].scatter(lead['y'][:,i], lead['z'][:,i], marker='o', s=1, linewidths=0, edgecolors='none', c= 'r', label='leading')
    axs[2].scatter(trail['y'][:,i], trail['z'][:,i], marker='o', s=1, linewidths=0, edgecolors='none', c= 'b', label='trailing')
    axs[2].plot(prog_orbit[i,1], prog_orbit[i,2], marker='*',markersize=5,color='k')
    axs[2].set_xlim(prog_orbit[i,1]-3, prog_orbit[i,1]+3)
    axs[2].set_ylim(prog_orbit[i,2]-3, prog_orbit[i,2]+3)
    axs[2].set_xlabel('y (kpc)')
    axs[2].set_ylabel('z (kpc)')
    axs[2].legend(loc=2)
    axs[2].add_patch(Ellipse(xy=(0,0),width=2,height=2))

    leadastro = coord.SkyCoord(x=lead['x'][:,i]*u.kpc,y=lead['y'][:,i]*u.kpc,z=lead['z'][:,i]*u.kpc,
                            v_x=lead['vx'][:,i]*u.km/u.s,v_y=lead['vy'][:,i]*u.km/u.s,v_z=lead['vz'][:,i]*u.km/u.s, frame="galactocentric")
    trailastro = coord.SkyCoord(x=trail['x'][:,i]*u.kpc,y=trail['y'][:,i]*u.kpc,z=trail['z'][:,i]*u.kpc,
                            v_x=trail['vx'][:,i]*u.km/u.s,v_y=trail['vy'][:,i]*u.km/u.s,v_z=trail['vz'][:,i]*u.km/u.s, frame="galactocentric")
    prog = coord.SkyCoord(x=prog_orbit[i,0]*u.kpc,y=prog_orbit[i,1]*u.kpc,z=prog_orbit[i,2]*u.kpc,
                            v_x=prog_orbit[i,3]*u.km/u.s,v_y=prog_orbit[i,4]*u.km/u.s,v_z=prog_orbit[i,5]*u.km/u.s, frame="galactocentric")
    
    axs[3].scatter(leadastro.icrs.ra.to_value(), leadastro.icrs.dec.to_value(), marker='o', s=1, linewidths=0, edgecolors='none', c= 'r')
    axs[3].scatter(trailastro.icrs.ra.to_value(), trailastro.icrs.dec.to_value(), marker='o', s=1, linewidths=0, edgecolors='none', c= 'b')
    axs[3].plot(prog.icrs.ra.to_value(), prog.icrs.dec.to_value(), marker='*',markersize=5,color='k')
    axs[3].set_xlim(prog.icrs.ra.to_value()-10, prog.icrs.ra.to_value()+10,)
    axs[3].set_ylim(prog.icrs.dec.to_value()-10, prog.icrs.dec.to_value()+10,)
    axs[3].set_xlabel(r'R.A. ($^\circ$)')
    axs[3].set_ylabel(r'Dec. ($^\circ$)')

    axs[4].scatter(leadastro.icrs.pm_ra_cosdec.to_value(), leadastro.icrs.pm_dec.to_value(), marker='o', s=1, linewidths=0, edgecolors='none', c= 'r')
    axs[4].scatter(trailastro.icrs.pm_ra_cosdec.to_value(), trailastro.icrs.pm_dec.to_value(), marker='o', s=1, linewidths=0, edgecolors='none', c= 'b')
    axs[4].plot(prog.icrs.pm_ra_cosdec.to_value(), prog.icrs.pm_dec.to_value(), marker='*',markersize=5,color='k')
    axs[4].set_xlim(prog.icrs.pm_ra_cosdec.to_value()-3, prog.icrs.pm_ra_cosdec.to_value()+3,)
    axs[4].set_ylim(prog.icrs.pm_dec.to_value()-3, prog.icrs.pm_dec.to_value()+3,)
    axs[4].set_xlabel(r'$\mu^*_\alpha$ (mas yr$^{-1}$)')
    axs[4].set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)')

    E_lead = MW_pot.potential(np.column_stack((lead['x'][:,i],lead['y'][:,i],lead['z'][:,i])), t=prog_timestamp[i]) +\
          0.5 * (lead['vx'][:,i]**2 + lead['vy'][:,i]**2 + lead['vz'][:,i]**2)
    E_trail = MW_pot.potential(np.column_stack((trail['x'][:,i],trail['y'][:,i],trail['z'][:,i])), t=prog_timestamp[i]) +\
          0.5 * (trail['vx'][:,i]**2 + trail['vy'][:,i]**2 + trail['vz'][:,i]**2)
    Lz_lead = lead['x'][:,i]*lead['vy'][:,i]-lead['y'][:,i]*lead['vx'][:,i]
    Lz_trail = trail['x'][:,i]*trail['vy'][:,i]-trail['y'][:,i]*trail['vx'][:,i]
    axs[5].scatter(Lz_lead, E_lead/1e5, marker='o', s=1, linewidths=0, edgecolors='none', c= 'r')
    axs[5].scatter(Lz_trail, E_trail/1e5, marker='o', s=1, linewidths=0, edgecolors='none', c= 'b')
    Lz_prog = prog_orbit[i,0]*prog_orbit[i,4]-prog_orbit[i,1]*prog_orbit[i,3]
    E_prog = MW_pot.potential(prog_orbit[i,0:3], t=prog_timestamp[i]) + 0.5 * np.sum(prog_orbit[i,3:]**2)
    axs[5].plot(Lz_prog, E_prog/1e5, marker='*',markersize=5,color='k')
    axs[5].set_xlim(Lz_prog-30, Lz_prog+30)
    axs[5].set_ylim((E_prog-1e3)/1e5, (E_prog+1e3)/1e5)
    axs[5].set_xlabel(r'L$_z$ (kpc km s$^{-1}$)',)
    axs[5].set_ylabel(r'Energy ($\times 10^5$ km$^2$ s$^{-2}$)', )
    axs[5].yaxis.set_label_position("right")
    axs[5].yaxis.tick_right()
    
    plt.suptitle('Axisymmetric Case', y=0.95)
    plt.draw()
    #plt.tight_layout()
    plt.pause(.01)

plt.ioff()
plt.show()

# %%
"""
# rewind
def icrs26d(x_data):
    mu = 14.58-0.2*(x_data['l']-5)
    cluster = coord.SkyCoord(ra= x_data['ra'] *u.degree, 
                            dec=  x_data['dec'] *u.degree,
                            distance= 10**(mu/5+1) *u.pc,
                            pm_ra_cosdec= x_data['pmra'] *u.mas/u.yr,
                            pm_dec= x_data['pmdec'] *u.mas/u.yr,
                            radial_velocity= x_data['vel_calib'] *u.km/u.s,  frame="icrs")
    cluster = cluster.galactocentric
    return np.column_stack((cluster.x.to_value(unit=u.kpc),cluster.y.to_value(unit=u.kpc),cluster.z.to_value(unit=u.kpc),
            cluster.v_x.to_value(unit=u.km/u.s),cluster.v_y.to_value(unit=u.km/u.s),cluster.v_z.to_value(unit=u.km/u.s),))

with fits.open("/Users/yyan0723/Desktop/S5/bar/ophiuchus_members_main.fits") as file:
    main_data = file[1].data
with fits.open("/Users/yyan0723/Desktop/S5/bar/ophiuchus_members_fan.fits") as file:
    fan_data = file[1].data
with fits.open("/Users/yyan0723/Desktop/S5/bar/ophiuchus_members_spur.fits") as file:
    spur_data = file[1].data

tend = -0.5/tunit
deltat = -1/tunit/1000
time = 0
main_traj = icrs26d(main_data)
fan_traj = icrs26d(fan_data)
spur_traj = icrs26d(spur_data)

axs = plt.subplots(1,3, figsize=[18,6],)[1].ravel()
plt.ion()
while np.round(time,4) > tend:
    for ax in axs:
        ax.cla()
    axs[0].plot(main_traj[:,0], main_traj[:,1], marker='.',color='r',markersize=3,linestyle="")
    axs[0].plot(fan_traj[:,0], fan_traj[:,1], marker='.',color='g',markersize=3,linestyle="")
    axs[0].plot(spur_traj[:,0], spur_traj[:,1], marker='.',color='b',markersize=3,linestyle="")
    axs[0].set_xlim(-15,15)
    axs[0].set_ylim(-15,15)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].text(-12,12,"{} Gyr".format(np.round(time*tunit,2)))
    axs[0].add_patch(Ellipse(xy=(0,0),width=4,height=2,angle=np.rad2deg(makeRotation(-27, -56, 0, time)[0,1]),))
    #axs[0].add_patch(Ellipse(xy=(0,0),width=2,height=2))

    axs[1].plot(main_traj[:,0], main_traj[:,2], marker='.',color='r',markersize=3,linestyle="")
    axs[1].plot(fan_traj[:,0], fan_traj[:,2], marker='.',color='g',markersize=3,linestyle="")
    axs[1].plot(spur_traj[:,0], spur_traj[:,2], marker='.',color='b',markersize=3,linestyle="")
    axs[1].set_xlim(-15,15)
    axs[1].set_ylim(-15,15)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    axs[1].text(-12,12,"r_gc= {} kpc".format(np.round(np.sum(np.median(main_traj[:,:3],axis=0)**2)**0.5, 2) ) )
    axs[1].add_patch(Ellipse(xy=(0,0),width=2,height=2))

    axs[2].plot(main_traj[:,1], main_traj[:,2], marker='.',color='r',markersize=3,linestyle="")
    axs[2].plot(fan_traj[:,1], fan_traj[:,2], marker='.',color='g',markersize=3,linestyle="")
    axs[2].plot(spur_traj[:,1], spur_traj[:,2], marker='.',color='b',markersize=3,linestyle="")
    axs[2].set_xlim(-15,15)
    axs[2].set_ylim(-15,15)
    axs[2].set_xlabel('y')
    axs[2].set_ylabel('z')
    axs[2].add_patch(Ellipse(xy=(0,0),width=2,height=2))
    plt.draw()
    plt.tight_layout()
    plt.pause(.1)

    main_traj = np.vstack(agama.orbit(ic= main_traj, potential=MW_pot, time=deltat, timestart=time, trajsize=1)[:,1])
    fan_traj = np.vstack(agama.orbit(ic= fan_traj, potential=MW_pot, time=deltat, timestart=time, trajsize=1)[:,1])
    spur_traj = np.vstack(agama.orbit(ic= spur_traj, potential=MW_pot, time=deltat, timestart=time, trajsize=1)[:,1])
    time = time + deltat

plt.ioff()
plt.show()
"""