import numpy as np
import scipy as sp
"""
Gradient-echo multi-echo combination (for phase).

Parameters:
    img: raw complex images from multiple receivers, 5D: [3D_image, echoes, receiver channels]
    vox: spatial resolution/voxel size, e.g. [1 1 1] for isotropic
    te: echo times
    mask: brain mask
    ph_cmb: phase after combination
    smooth_method: smooth methods(1) smooth3, (2) poly3, (3) poly3_nlcg, (4) gaussian
    parpool_flag:

Returns:
    ph_cmb:
    mag_cmb:
    coil_sens
"""
def geme_cmb(img, vox, te, 
             mask, 
             smooth_method="gaussian", 
             parpool_flag=0):

    ne = np.shape(img)[-1] # Number of echos
    nrcvrs  = 1 # Number of recievers
    TE0 = te[0]
    TE1 = te[1]
    imsize = np.shape(img)

    np.seterr(divide='ignore')
    img_diff = np.divide(img[:,:,:,2], img[:,:,:,1]) 
    ph_diff = np.divide(img_diff, np.abs(img_diff))
    ph_diff_cmd = np.sum(np.abs(img[:,:,:,1])*ph_diff)
    print(ph_diff_cmd)