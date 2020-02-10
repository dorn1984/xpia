import numpy as np
import os, xarray
import numpy.matlib
from scipy import fftpack
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
fs=14
plt.rc("font",size=fs)

def get_psd_2d(array, spacing):

    (npts,npts) = array.shape

    # apply window (to minimize noise since the arrays are not periodic in space)
    hamm_2d = return_hamming(Nx=npts,Ny=npts)    
    array_windowed = array * hamm_2d

    # apply fft
    fftarray  = fftpack.fft2(array_windowed)
    
    # get psd
    fftarray_shifted = fftpack.fftshift(fftarray)
    psd_2d   = (fftarray_shifted[1:,1:] * np.conj(fftarray_shifted[1:,1:])).real
    
    # get frequencies
    fr = fftpack.fftfreq(npts, d=spacing)
    fr = fr[1:]
    fr = np.sort(fr)
    
    [fr_x,fr_y] = np.meshgrid(fr,fr)
    
    f1d = fr
    f2d = fr_x
    
    return f1d, f2d, psd_2d


def get_csd_2d(array1, array2, spacing):

    (npts,npts) = array1.shape

    # apply window (to minimize noise since the arrays are not periodic in space)
    hamm_2d = return_hamming(Nx=npts,Ny=npts)    
    array1_windowed = array1 * hamm_2d
    array2_windowed = array2 * hamm_2d    

    # apply fft
    fftarray1  = fftpack.fft2(array1_windowed)
    fftarray2  = fftpack.fft2(array2_windowed)    
    
    # get csd
    fftarray1_shifted = fftpack.fftshift(fftarray1)
    fftarray2_shifted = fftpack.fftshift(fftarray2)    
    csd_2d   = (np.conj(fftarray1_shifted[1:,1:])*fftarray2_shifted[1:,1:]).real
    
    # get frequencies
    fr = fftpack.fftfreq(npts, d=spacing)
    fr = fr[1:]
    fr = np.sort(fr)
    
    [fr_x,fr_y] = np.meshgrid(fr,fr)
    
    f1d = fr
    f2d = fr_x
    
    return f1d, f2d, csd_2d

def get_prime_and_means(datapath, datetime, zi, varname='w', z_zi_target=0.5, sims=["LES","vles","mynn","ysu","sh"], verbose=False):
    
    datetime_str = "{0:%Y-%m-%d_%H:%M}".format(datetime)
    
    xy_primes    = {}
    xy_means     = {}    
    
    for sim in sims:        

        sim_prefix      = sim+"_25m" if sim=="LES" else sim
        fpath           = os.path.join(datapath,"WRF_{0}_3D_{1}_plus_filtered.nc".format(sim_prefix,datetime_str))    
        data            = xarray.open_dataset(fpath)
        
        if varname=='ws':
            data["ws_filt"] = np.sqrt(data["u_filt"]**2+data["v_filt"]**2)
            data["ws"]      = np.sqrt(data["u"]**2+data["v"]**2)    

        z         = data['z'].copy()
        zmean     = np.median(z.values,axis=(1,2))
        z_zi      = zmean/zi    
        k         = np.argmin(np.abs(z_zi-z_zi_target))    

        variable  = data[varname+"_filt"].copy()
        variable  = variable.values[k,...].copy()
    
        xy_mean   = np.mean(variable)
        xy_prime  = variable - xy_mean
        
        xy_primes[sim] = xy_prime.copy()
        xy_means[sim]  = xy_mean
        
        if verbose:
            print("................")            
            print("Simulation : {0}".format(sim))
            print("File : {0}".format(os.path.split(fpath)[-1]))
            print("Vertical level : {0}".format(k)) 
            print("Getting {0}".format(varname+"_filt"))

        # sim == "LES" also get the raw values, i.e. without the primes
        if sim=="LES":
            if verbose:            
                print("Getting {0}".format(varname))        
            variable              = data[varname].copy()
            variable              = variable.values[k,...]           
            xy_mean               = np.mean(variable)
            xy_prime              = variable - xy_mean
            xy_primes[sim+"_raw"] = xy_prime.copy()    
            xy_means[sim+"_raw"]  = xy_mean
            
    return xy_primes, xy_means

def coarsen(dictionary_of_arrays, target_delta=333.0, target_npts=90):
    
    dictionary_coarse = {}
    
    x1d_desired = np.arange(0,target_npts*target_delta,target_delta)
    y1d_desired = x1d_desired.copy()    
    
    for sim in dictionary_of_arrays.keys():    
        npts, npts = dictionary_of_arrays[sim].shape
        if npts>500:
            delta      = 25.0
            x1d        = np.arange(0,npts*delta,delta)
            y1d        = x1d.copy()                
            func       = RectBivariateSpline(x1d, y1d, dictionary_of_arrays[sim])            
            dictionary_coarse[sim] = func(x1d_desired, y1d_desired)        
        else:
            dictionary_coarse[sim] = dictionary_of_arrays[sim].copy()
            
    return dictionary_coarse

def make_finer(dictionary_of_arrays, target_delta=25.0, target_npts=1200):
    
    dictionary_coarse = {}
    
    x1d_desired = np.arange(0,target_npts*target_delta,target_delta)
    y1d_desired = x1d_desired.copy()    
    
    for sim in dictionary_of_arrays.keys():    
        npts, npts = dictionary_of_arrays[sim].shape
        if npts<500:
            delta      = 333.0
            x1d        = np.arange(0,npts*delta,delta)
            y1d        = x1d.copy()                
            func       = RectBivariateSpline(x1d, y1d, dictionary_of_arrays[sim])            
            dictionary_coarse[sim] = func(x1d_desired, y1d_desired)        
        else:
            dictionary_coarse[sim] = dictionary_of_arrays[sim].copy()
            
    return dictionary_coarse

def psd_cartesian_to_polar(f1d, psd_2d, thetas=None, radii_wavelength=None, min_radius=50.0, max_radius=30000.0):

    """
    Parameters
    ----------
    radii_wavelength : np.array,
        one-dimensional array of desired radii [m]
    thetas : np.array,
        one-dimensional array of desired azimuths [radians]
    """
    
    f = RectBivariateSpline(f1d,f1d,psd_2d) 
        
    if radii_wavelength is None:
        radii_wavelength = np.append(radii_wavelength, np.arange(min_radius,3000,10))
        radii_wavelength = np.append(radii_wavelength, np.arange(3000,max_radius+0.1,250))

    radii_wavenumber = 1/radii_wavelength    
    nr               = len(radii_wavelength)
    
    if thetas is None:
        thetas           = np.radians(np.arange(135,315.1,1))
    else:
        if np.any(thetas>2.1*np.pi):
            thetas = np.radians(thetas)
    ntheta           = len(thetas)

    theta_polar      = np.zeros((nr,ntheta))
    x_polar          = np.zeros((nr,ntheta))
    y_polar          = np.zeros((nr,ntheta))
    psd_polar        = np.zeros((nr,ntheta))
    radii_polar      = np.zeros((nr,ntheta))

    for ir,rr in enumerate(radii_wavenumber):
        for itheta,theta in enumerate(thetas):
            xx  = rr*np.cos(theta)
            yy  = rr*np.sin(theta)         
            val = f(xx,yy)[0][0]
            
            val = np.nan if val<0 else val

            x_polar[ir,itheta]     = xx
            y_polar[ir,itheta]     = yy
            psd_polar[ir,itheta]   = val
            theta_polar[ir,itheta] = theta
            radii_polar[ir,itheta] = rr
            
    return x_polar, y_polar, theta_polar, radii_polar, psd_polar

def plot_psd_polar(theta, radii, psd_2d_polar,log10=True,vmin=None,vmax=None):
    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111,projection='polar')
    
    if log10:
        contourvals=np.log10(psd_2d_polar)
    else:
        contourvals=psd_2d_polar
    
    if vmin is None:
        p = ax.contourf(theta, radii, contourvals)
    else:
        p = ax.contourf(theta, radii, contourvals,levels=np.arange(vmin,vmax+0.1,1))
    plt.colorbar(p)
    
    ax.set_xlim([np.min(theta), np.max(theta)])
    return fig

def plot_psd_cartesian(f1d,psd_2d,log10=True,vmin=None,vmax=None):
    xticks_wavelength = [-30000,-100,-50,50,100]
    xticks_wavenum    = [1/a for a in xticks_wavelength]
    xticks_labels     = ["{0:.0f}".format(a) for a in xticks_wavelength]
    idx_center        = np.where(np.abs(np.array(xticks_wavelength))>20000)[0][0]
    xticks_labels[idx_center] = "+-30000"

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111,aspect="equal")

    if log10:
        contourvals = np.log10(psd_2d)
    else:
        contourvals = psd_2d.copy()

    if vmin is None:
        p = ax.contourf(f1d, f1d, contourvals)
    else:
        p = ax.contourf(f1d, f1d, contourvals,levels=np.arange(vmin,vmax+0.1,1))

    ax.set_xticks(xticks_wavenum)
    ax.set_xticklabels(xticks_labels)
    ax.set_xlabel("wavelength $\lambda_x$ [m]")

    ax.set_yticks(xticks_wavenum)
    ax.set_yticklabels(xticks_labels)
    ax.set_ylabel("wavelength $\lambda_y$ [m]")

    plt.colorbar(p)
    
    return fig

def return_hamming(Nx,Ny):

    # combo of uniform window in the middle and hamming window on edges:
    npts_buffer = 16

    # create a 1-d hamming window : hamm_1d.shape = (100,)
    hamm_1d     = np.hamming(npts_buffer*2)
    # repeat it : B.shape = (100, 100)
    B           = np.matlib.repmat(hamm_1d,npts_buffer*2,1)
    # transpose it : C.shape = (100,100)
    C           = np.transpose(B)
    # now get the two-dimensional hamming window : hamm_2d_s.shape = (100,100)
    hamm_2d_s   = B*C

    # allocate space for the final 2d-filter
    hamm_2d = np.zeros([Ny,Nx])

    # fill it with ones (no filter) anywhere that the window won't be applied (inside the domain, anywhere inside the buffer)        
    for ii in range(0+npts_buffer,Nx-npts_buffer):
        for jj in range(0+npts_buffer,Ny-npts_buffer):
            hamm_2d[jj,ii] = 1.0

    # now put the filter values in there

    # south west corner
    hamm_2d[0:npts_buffer,0:npts_buffer] = hamm_2d_s[0:npts_buffer,0:npts_buffer]

    # south east corner
    hamm_2d[0:npts_buffer,Nx-npts_buffer:Nx] = hamm_2d_s[0:npts_buffer,2*npts_buffer-npts_buffer:2*npts_buffer]

    # north west corner
    hamm_2d[Ny-npts_buffer:Ny,0:npts_buffer] = hamm_2d_s[2*npts_buffer-npts_buffer:2*npts_buffer,0:npts_buffer]

    # north east corner
    hamm_2d[Ny-npts_buffer:Ny,Nx-npts_buffer:Nx] = hamm_2d_s[2*npts_buffer-npts_buffer:2*npts_buffer,2*npts_buffer-npts_buffer:2*npts_buffer]

    # south boundary
    hann_tmp = hamm_1d[0:npts_buffer]
    len_tmp  = Nx-2*npts_buffer
    hann_tmp = np.matlib.repmat(hann_tmp,len_tmp,1)
    hann_tmp = np.transpose(hann_tmp)
    hamm_2d[0:npts_buffer,npts_buffer:Nx-npts_buffer] = hann_tmp

    # north boundary
    hann_tmp = hamm_1d[npts_buffer:2*npts_buffer]
    len_tmp  = Nx-2*npts_buffer
    hann_tmp = np.matlib.repmat(hann_tmp,len_tmp,1)
    hann_tmp = np.transpose(hann_tmp)
    hamm_2d[Ny-npts_buffer:Ny,npts_buffer:Nx-npts_buffer] = hann_tmp

    # west boundary
    hann_tmp = hamm_1d[0:npts_buffer]
    len_tmp  = Ny-2*npts_buffer
    hann_tmp = np.matlib.repmat(hann_tmp,len_tmp,1)
    hamm_2d[npts_buffer:Ny-npts_buffer,0:npts_buffer] = hann_tmp

    # east boundary
    hann_tmp = hamm_1d[npts_buffer:2*npts_buffer]
    len_tmp  = Ny-2*npts_buffer
    hann_tmp = np.matlib.repmat(hann_tmp,len_tmp,1)
    hamm_2d[npts_buffer:Ny-npts_buffer,Nx-npts_buffer:Nx] = hann_tmp
    
    return hamm_2d
