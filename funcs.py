import numpy as np
import numpy.matlib
from scipy import fftpack

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
