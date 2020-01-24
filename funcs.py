import numpy as np
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
    psd_2d   = np.abs(fftarray_shifted[1:,1:])**2
    
    # get frequencies
    fr = fftpack.fftfreq(npts, d=spacing)
    fr = fr[1:]
    fr = np.sort(fr)
    
    [fr_x,fr_y] = np.meshgrid(fr,fr)
    
    f1d = fr
    f2d = fr_x
    
    return f1d, f2d, psd_2d

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
        radii_wavelength = np.arange(min_radius,300,1) 
        radii_wavelength = np.append(radii_wavelength, np.arange(300,3000,20))
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
