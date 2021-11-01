#Stitch input,output and ground truth imgs together for comparison

import numpy as np
import glob
from PIL import Image, ImageFont, ImageDraw
import random
import parser
import os
import scipy.ndimage as ndimage
from skimage import io

def PSNR(I0,I1):
    MSE = np.mean( (I0-I1)**2, dtype = 'float32')
    PSNR = 20*np.log10(255/np.sqrt(MSE), dtype ='float32')
    return PSNR

def stitcher(inputdir,outputdir,gtdir,stitchdir):
    #n ~ noisy, dn ~ denoised, gt~ ground truth file list
    n = sorted(glob.glob('%s/*.png' % inputdir))
    dn = sorted(glob.glob('%s/*.png' % outputdir))
    gt = sorted(glob.glob('%s/*.png' % gtdir))
    

    try:
        os.mkdir(stitchdir)

    except OSError:
        pass

    for i in range(0,len(n)):

    #open images and stitch sequentially
        noisy = Image.open(n[i])
        noisy = np.array(noisy)
        
        denoised = Image.open(dn[i])
        denoised = np.array(denoised)

        truth = Image.open(gt[i])
        truth = np.array(truth)

        stitched = np.concatenate((noisy,denoised,truth),axis=1)
        filename = '%s/%d.png' % (stitchdir,i)
        io.imsave(filename,stitched)

        quality1 = PSNR(noisy,truth)
        quality2 = PSNR(denoised, truth)
        caption_text1 = "PSNR=%.2fdB" % (quality1)
        caption_text2 = "PSNR=%.2fdB" % (quality2)

        caption_font = ImageFont.truetype('IBM_Plex_Sans/IBMPlexSans-SemiBold.ttf',size=20)
        gt_text = "Ground truth"
        gt_font = ImageFont.truetype('IBM_Plex_Sans/IBMPlexSans-SemiBoldItalic.ttf',size=20)
        
        simg = Image.open(filename)
        stitched = ImageDraw.Draw(simg)
        stitched.text((15,15), caption_text1, font = caption_font)
        stitched.text((527,15), caption_text2, font = caption_font)
        stitched.text((1039,15), gt_text, font = gt_font)
     
        simg.save(filename)


#---------------------------------------------------------------
inputdir = 'testdata/input/noisy_512'
outputdir = 'testdata/output'
gtdir = 'testdata/ground_truth/noisy_512'
stitchdir ='testdata/stitched/noisy_512_v1.2'

stitcher(inputdir,outputdir,gtdir,stitchdir)



