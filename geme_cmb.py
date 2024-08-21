import numpy as np
import scipy as sp
import nibabel as nib
from numpy import shape
import os
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
    img = np.expand_dims(img, axis=4)
    img_diff = np.divide(img[:,:,:,1,:], img[:,:,:,0,:]) 
    ph_diff = np.divide(img_diff, np.abs(img_diff))

    a = np.multiply(np.abs(img[:,:,:,0]), ph_diff)
    ph_diff_cmb = np.sum(a, axis=3)
    ph_diff_cmb[np.isnan(ph_diff_cmb)] = 0
    
    nii = nib.Nifti1Image(np.angle(ph_diff_cmb), affine=np.eye(4))
    nii.to_filename("output/ph_diff.nii")
    
    mag1 = np.sqrt(np.sum(np.abs(img[:,:,:,0,:]**2),axis=3))
    mask_input = mask
    a = mask.astype(bool)
    b = np.median(mag1[a])
    c = np.multiply(0.1, b)
    mask = (mag1 > c)
    mask = np.logical_or(mask, mask_input.astype(bool))
    
    NV, NP, NS = imsize[0:3]

    with open('output/wrapped_phase_diff.dat', 'wb') as f:
        np.angle(ph_diff_cmb).tofile(f)

    mask_unwrp = (255 * mask).astype(np.uint8)
    with open('output/mask_unwrp.dat', 'wb') as f:
        mask_unwrp.tofile(f)

    command = 'phase_unwrapping/3DSRNCP ' \
                'output/wrapped_phase_diff.dat ' \
                'output/mask_unwrp.dat ' \
                'output/unwrapped_phase_diff.dat '\
                '{} {} {} output/reliability_diff.dat'.format(NV, NP, NS)
    os.system(command)