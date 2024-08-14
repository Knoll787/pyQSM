import numpy as np
import scipy as sp
import os
import pydicom 
import nibabel as nib
from geme_cmb import geme_cmb


def main():
    # Default Values
    readout = 'unipolar'
    r_mask = 1
    fit_thr = 40
    bet_thr = 0.4
    bet_smooth = 2
    ph_unwrap = 'bestpath'
    bkg_rm = 'resharp'
    t_svd = 0.1
    smv_rad = 3
    tik_reg = 1e-3
    cgs_num = 500
    lbv_tol = 0.01
    lbv_peel = 2
    tv_reg = 5e-4
    inv_num = 500
    interp = 0

    #
    # Import Magnitude DICOMs
    #
    path_mag = "data/magnitude/"
    mag_list = sorted(os.listdir(path_mag))
    
    # Get sequence parametres
    dicom_info = pydicom.dcmread(path_mag + mag_list[-1])
    numImages = len(mag_list) # Total number of images in the directory
    echoTrainLength = int(dicom_info.EchoNumbers) # Determines the number of different TEs in a directory
    sequenceLength = int(numImages / echoTrainLength) # Number of images in each scan of a given TE
    imageWidth = dicom_info.Columns
    imageHeight = dicom_info.Rows

    # Determine the different TE values used for each scan
    TE = []
    for i in range(0, numImages, int(numImages/echoTrainLength)):
        dicom_info = pydicom.dcmread(path_mag + mag_list[i])
        TE.append(dicom_info.EchoTime * 1e-3)
    vox = [dicom_info.PixelSpacing[0], dicom_info.PixelSpacing[1], dicom_info.SliceThickness]
    

    # Projections in the Z-axis
    Xz = dicom_info.ImageOrientationPatient[2];
    Yz = dicom_info.ImageOrientationPatient[5];
    Zxyz = np.cross(dicom_info.ImageOrientationPatient[0:3], dicom_info.ImageOrientationPatient[3:6])
    Zz = Zxyz[2];
    z_prjs = [Xz, Yz, Zz];
    
    # Generate array of ALL Magnitude image data
    mag = np.zeros([imageWidth, imageHeight, sequenceLength, echoTrainLength])
    for i in range(0, len(mag_list)):
        a = pydicom.dcmread(path_mag + mag_list[i]).pixel_array
        x = [int(len(mag_list)/echoTrainLength), echoTrainLength]
        NS, NE = np.unravel_index(i, x, order="F") # NS -> Number Scan, NE -> Number Echo
        imageData = np.transpose(a, (1,0))
        mag[:,:, NS, NE] = imageData 
    imsize = np.shape(mag)
    
    # Import Phase data
    path_ph= "data/phase/"
    ph_list = sorted(os.listdir(path_ph))

    # Generate array of ALL Phase image data
    ph = np.empty([imageWidth, imageHeight, sequenceLength, echoTrainLength])
    for i in range(0, len(ph_list)):
        a = pydicom.dcmread(path_ph + ph_list[i])
        a = a.pixel_array
        x = [int(len(ph_list)/echoTrainLength), echoTrainLength]
        NS, NE = np.unravel_index(i, x, order="F") # NS -> Number Scan, NE -> Number Echo
        imageData = np.transpose(a, (1,0))
        ph[:,:, NS, NE] = imageData / 4095*2*np.pi - np.pi

    # Setup output directory
    os.makedirs("output/", exist_ok=True)
    
    # NIfTI Generation
    os.makedirs("output/src", exist_ok=True)
    for i in range(0, echoTrainLength):
        # Magnitude -> NIfTI
        nii = nib.Nifti1Image(mag[:,:,:,i], affine=np.eye(4))
        nii.to_filename("output/src/mag_" + str(i) + ".nii")

        # Phase -> NIfTI
        nii = nib.Nifti1Image(ph[:,:,:,i], affine=np.eye(4))
        nii.to_filename("output/src/ph_" + str(i) + ".nii")
        

    # Brain Extraction
    # Generate mask from magnitude of the 1st Echo -> Using FSL tools       
    print("Extracting brain volume and generating mask...")
    os.system('rm BET*')
    command = '~/fsl/bin/bet2 output/src/mag_0.nii ./output/BET -f {} -m -w {}'.format(bet_thr, bet_smooth)
    os.system(command)
    os.system('gunzip -f output/BET.nii.gz') # Masking file 
    os.system('gunzip -f output/BET_mask.nii.gz') # Isolated brain

    mask_img = nib.load('output/BET_mask.nii')
    mask = mask_img.get_fdata()

    
    geme_cmb(mag*np.exp(1j*ph), vox, TE, mask)


if __name__ == "__main__":
    main()