import glob
from PIL import Image
import random
import parser
import os
import scipy.ndimage as ndimage

import numpy as np
import pickle

from skimage.measure import compare_ssim
from skimage import io


def blur(src,OTF):
    #current code assumes uint8 synthetic one channel OTF
    otf_img = Image.open(OTF)
    otf = np.array(otf_img)
    #otf_img = np.sum(otf_img,2)/765
    img_spec = np.fft.fft2(src)
    img_spec = np.fft.fftshift(img_spec)
    clipped_spectrum = img_spec * (otf/255)
    blurred = np.fft.ifftshift(clipped_spectrum)
    blurred = np.fft.ifft2(blurred)
    print(np.max(blurred))
    return abs(blurred)
    


def partitionDataset(imgs,outdir,nreps,dim,degradeBool=True):
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    for i in range(0,len(imgs)):
        
        #open image and greyscale
        src_img = Image.open(imgs[i])
        src_img = np.array(src_img)
        gs_img = np.sum(src_img, 2)/3
     

        h,w = gs_img.shape[0:2]

        j = 0
        while j < nreps:
            # random cropping 
            r_rand = np.random.randint(0,h-dim)
            c_rand = np.random.randint(0,w-dim)
            img = gs_img[r_rand:r_rand+dim,c_rand:c_rand+dim]
            #img is a uint8, make a copy
            gt_img = img.copy()


            # adding blur
            img = blur(img,'OTF.png')

            
            filename = '%s/%d-%d.pkl' % (outdir,i,j)

            print(i,j,r_rand,c_rand,img.shape,gt_img.shape)

            img = Image.fromarray(img.astype('uint8'))
            gt_img = Image.fromarray(gt_img.astype('uint8'))
            pickle.dump((img,gt_img), open(filename,'wb'))

            combined = np.concatenate((np.array(img),np.array(gt_img)),axis=1)
            io.imsave(filename.replace(".pkl",".png"),combined)

            j += 1

        print('[%d/%d]' % (i+1,len(imgs)))


# --------------------------------------------

nreps = 3
dim = 128

allimgs = sorted(glob.glob('DIV2K_train_HR/*.png'))[100:800]
outdir = 'trainingdata/noisy_' + str(dim)
print('Training data')

partitionDataset(allimgs,outdir,nreps,dim,False)
