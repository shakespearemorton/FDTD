import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep.materials import Au
from meep.materials import aSi
from meep.materials import Ti


#----------------------------Variables------------------------------
period = 1.5
THICKNESS = 0.045
PADDING = 2 							#um
fmin = 1/1.2							#maximum wavelength
fmax = 1/.300 							#minimum wavelength
fcen = (fmax+fmin)/2 						#set centre of gaussian source
df = fmax-fmin 							#set width of gaussian source
nfreq = 50 								#number of frequencies between min and max
dpml = 0.4 							#thickness of PML (top and bottom) um
resolution = 300 							#pixels/um
BASE = PADDING-THICKNESS 						#metal thin film is set on a PDMS base
sy = THICKNESS + 2 * PADDING + 2 * dpml 				#size of simulation
sx = period 							#size of simulation

box = 0.010 							#optimised size of radiation monitoring box set around dipole
cell = mp.Vector3(sx, sy, 0)

runtime = 800

#define geometry
slab = mp.Block(size=mp.Vector3(1e20, THICKNESS), center=mp.Vector3(0, -THICKNESS/2), material=Au) 			#Gold thin film
slab3 = mp.Block(mp.Vector3(1e20, BASE), center=mp.Vector3(0, -0.5*sy+dpml+0.5*BASE), material=aSi)	#pdms base
slab2 = mp.Block(size=mp.Vector3(1e20, 0.002), center=mp.Vector3(0, 0.001), material=Ti) 			#Gold thin film
geometry = [slab3,slab,slab2]

#--------------------------Simulation Parameters----------------------------------------

#define Gaussian plane wave Ez polarised
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ex,
        center=mp.Vector3(0, 0.5*sy-dpml-0.01),
        size=mp.Vector3(period,0),
    )
]

#define pml layers (Absorber is a type of PML that helps when there are Bloch Wave SPP modes. Placed in substrate)
pml_layers = [mp.PML(thickness=dpml, direction = mp.Y)]

#sets the simulation without the substrate so that it is a homogeneous environment
sim = mp.Simulation(cell_size=cell,
        boundary_layers=pml_layers,
        sources=sources,
        resolution=resolution,
        k_point=mp.Vector3(0,0,0))

#power monitors around the simulation
#power monitor around the dipole
tran_fr = mp.FluxRegion(center=mp.Vector3(0,-THICKNESS), size=mp.Vector3(x = sx) )
refl_fr = mp.FluxRegion(center=mp.Vector3(0,THICKNESS), size=mp.Vector3(x = sx),weight=-1)
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran = sim.add_flux(fcen,df, nfreq, tran_fr)
#run simulation until the source has decayed "fully"
sim.run(until_after_sources=runtime)

#collect radiation information
init_refl_data = sim.get_flux_data(refl)
init_tran_flux = mp.get_fluxes(tran)
sim.reset_meep()

#run simulation again with the substrate (inhomogeneous environment)
sim = mp.Simulation(cell_size=cell,
        boundary_layers=pml_layers,
        sources=sources,
        geometry=geometry,
        resolution=resolution,
        k_point=mp.Vector3(0,0,0))

#power monitors around the dipole
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

sim.run(until_after_sources=runtime)

final_refl_flux = mp.get_fluxes(refl)
final_tran_flux = mp.get_fluxes(tran)
flux_freqs = mp.get_flux_freqs(refl)

wl = []
Rs = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, 1/flux_freqs[i])
    Rs = np.append(Rs,-final_refl_flux[i]/init_tran_flux[i])
    Ts = np.append(Ts,final_tran_flux[i]/init_tran_flux[i])
As = 1-Rs-Ts
Data = np.zeros((len(wl),4))
Data[:,0]=wl
Data[:,1]=Rs
Data[:,2]=Ts
Data[:,3]=As

if mp.am_master():
    plt.figure()
    plt.plot(wl,Rs,'bo-',label='absorption')
    plt.plot(wl,Ts,'ro-',label='transmittance')
    plt.plot(wl,As,'go-',label='reflection')
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.savefig('Extinction.png')
    
np.savetxt('data.txt',Data)
