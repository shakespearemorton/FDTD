import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep.materials import Au

radii = [25]
for r in radii:
    r =r/1000
    wvl_min = 0.300
    wvl_max = 0.700

    frq_min = 1/wvl_max
    frq_max = 1/wvl_min
    frq_cen = 0.5*(frq_min+frq_max)
    dfrq = frq_max-frq_min
    nfrq = 350

    ## at least 8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
    resolution = 400

    dpml = 0.5*wvl_max
    dair = 0.5*wvl_max

    pml_layers = [mp.PML(thickness=dpml)]

    s = 2*(dpml+dair+r)
    cell_size = mp.Vector3(s,s,s)

    src_cmpt = mp.Hz
    sx=s
    sy=s
    sz=s
    # is_integrated=True necessary for any planewave source extending into PML
    sources = [mp.Source(mp.GaussianSource(frq_cen,fwidth=dfrq,is_integrated=True),
                         center=mp.Vector3(0,0.5*s-dpml,0),
                         size=mp.Vector3(s,0,s),
                         component=src_cmpt)]

    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3())

    #trans = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-0.5*s+dpml),size=mp.Vector3(4*r,0,4*r)))
    top = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=2*r),size=mp.Vector3(4*r,0,4*r)))
    bottom = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-2*r),size=mp.Vector3(4*r,0,4*r)))
    left = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-2*r),size=mp.Vector3(0,4*r,4*r)))
    right = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=2*r),size=mp.Vector3(0,4*r,4*r)))
    front = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=2*r),size=mp.Vector3(4*r,4*r,0)))
    back = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-2*r),size=mp.Vector3(4*r,4*r,0)))


    sim.run(until_after_sources=mp.stop_when_fields_decayed(5,src_cmpt,mp.Vector3(0,-0.5*s+dpml,0),1e-6))

    incident = mp.get_fluxes(top)
    freqs = mp.get_flux_freqs(top)
    top_data = sim.get_flux_data(top)
    bottom_data = sim.get_flux_data(bottom)
    left_data = sim.get_flux_data(left)
    right_data = sim.get_flux_data(right)
    front_data = sim.get_flux_data(front)
    back_data = sim.get_flux_data(back)

    sim.reset_meep()


    geometry = [mp.Sphere(radius=r, center=mp.Vector3(0,0,0),material=Au)]

    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        geometry=geometry)

    top = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=2*r),size=mp.Vector3(4*r,0,4*r)))
    bottom = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-2*r),size=mp.Vector3(4*r,0,4*r)))
    left = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-2*r),size=mp.Vector3(0,4*r,4*r)))
    right = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=2*r),size=mp.Vector3(0,4*r,4*r)))
    front = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=2*r),size=mp.Vector3(4*r,4*r,0)))
    back = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-2*r),size=mp.Vector3(4*r,4*r,0)))

    sim.load_minus_flux_data(top, top_data)
    sim.load_minus_flux_data(bottom, bottom_data)
    sim.load_minus_flux_data(left, left_data)
    sim.load_minus_flux_data(right, right_data)
    sim.load_minus_flux_data(front, front_data)
    sim.load_minus_flux_data(back, back_data)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(5,src_cmpt,mp.Vector3(0,-0.5*s+dpml,0),1e-6))

    top_flux = mp.get_fluxes(top)
    bottom_flux = mp.get_fluxes(bottom)
    left_flux = mp.get_fluxes(left)
    right_flux = mp.get_fluxes(right)
    front_flux = mp.get_fluxes(front)
    back_flux = mp.get_fluxes(back)


    intensity = np.asarray(incident)
    top = np.asarray(top_flux)
    bottom = np.asarray(bottom_flux)
    left = np.asarray(left_flux)
    right = np.asarray(right_flux)
    front = np.asarray(front_flux)
    back = np.asarray(back_flux)
    
    data = np.zeros((8,len(intensity)))
    data[0,:] = freqs
    data[1,:] = intensity
    data[2,:] = top
    data[3,:] = bottom
    data[4,:] = left
    data[5,:] = right
    data[6,:] = front
    data[7,:] = back

    namer = int(r*1000)

    np.savetxt(repr(namer)+'_scatt.txt',data)
    
    
    sim.reset_meep()
    
    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        geometry=geometry)

    top = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=2*r),size=mp.Vector3(4*r,0,4*r)))
    bottom = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-2*r),size=mp.Vector3(4*r,0,4*r)))
    left = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-2*r),size=mp.Vector3(0,4*r,4*r)))
    right = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=2*r),size=mp.Vector3(0,4*r,4*r)))
    front = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=2*r),size=mp.Vector3(4*r,4*r,0)))
    back = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-2*r),size=mp.Vector3(4*r,4*r,0)))

    sim.run(until_after_sources=mp.stop_when_fields_decayed(5,src_cmpt,mp.Vector3(0,-0.5*s+dpml),1e-6))

    top_flux = mp.get_fluxes(top)
    bottom_flux = mp.get_fluxes(bottom)
    left_flux = mp.get_fluxes(left)
    right_flux = mp.get_fluxes(right)
    front_flux = mp.get_fluxes(front)
    back_flux = mp.get_fluxes(back)


    intensity = np.asarray(incident)
    top = np.asarray(top_flux)
    bottom = np.asarray(bottom_flux)
    left = np.asarray(left_flux)
    right = np.asarray(right_flux)
    front = np.asarray(front_flux)
    back = np.asarray(back_flux)

    data = np.zeros((8,len(intensity)))
    data[0,:] = freqs
    data[1,:] = intensity
    data[2,:] = top
    data[3,:] = bottom
    data[4,:] = left
    data[5,:] = right
    data[6,:] = front
    data[7,:] = back

    np.savetxt(repr(namer)+'_abs.txt',data)
    
    data = np.loadtxt(repr(i)+'_scatt.txt')
    scatt = abs(data[4,:]-data[5,:]-data[2,:]+data[3,:]-data[6,:]+data[7,:])
    rad = i*10**-9
    sx = (4*rad)**2
    inc = abs(data[1,:])/sx
    scatt = scatt/inc
    plt.plot(1000*data[0,:]**-1,scatt,label = 'scattering')
    data = np.loadtxt(repr(i)+'_abs.txt')
    absorp = abs(data[4,:]-data[5,:]-data[2,:]+data[3,:]-data[6,:]+data[7,:])
    absorp = absorp/inc
    plt.plot(1000*data[0,:]**-1,absorp,label = 'absorption')
    plt.plot(1000*data[0,:]**-1,scatt+absorp,label = 'extinction')
    
    
plt.legend()
plt.savefig('extinction_spectra.png')
