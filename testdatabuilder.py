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
    


def partitionTestset(imgs,imgoutdir,gtoutdir,nreps,dim,degradeBool=True):
    try:
        os.makedirs(imgoutdir)
        os.makedirs(gtoutdir)
    except OSError:
        pass

    for i in range(0,len(imgs)):
        
        src_img = Image.open(imgs[i])
        src_img = np.array(src_img)

         #open image and greyscale
        src_img = Image.open(imgs[i])
        src_img = np.array(src_img)
        gs_img = np.sum(src_img, 2)/3
                 
   
        h,w = src_img.shape[0:2]

        j = 0
        while j < nreps:

             # random cropping 
            r_rand = np.random.randint(0,h-dim)
            c_rand = np.random.randint(0,w-dim)
            img = gs_img[r_rand:r_rand+dim,c_rand:c_rand+dim]
            #img is a uint8, make a copy
            gt_img = img.copy()


            # adding blur
            img = blur(img,'OTF_512.png')
            
            
            filename = '%s/%d-%d_testimg.png' % (imgoutdir,i,j)
            gtfilename = '%s/%d-%d.png' % (gtoutdir,i,j)

            print(i,j,r_rand,c_rand,img.shape,gt_img.shape)

            img = Image.fromarray(img.astype('uint8'))
            gt_img = Image.fromarray(gt_img.astype('uint8'))
            pickle.dump((img,gt_img), open(filename,'wb'))

            io.imsave(filename,np.array(img))
            io.imsave(gtfilename,np.array(gt_img))


            j += 1

        print('[%d/%d]' % (i+1,len(imgs)))


# --------------------------------------------

nreps = 3
#dim = 128
dim = 512

allimgs = [
    "DIV2K_train_HR/0031.png",
    "DIV2K_train_HR/0032.png",
    "DIV2K_train_HR/0033.png",
    "DIV2K_train_HR/0034.png",
    "DIV2K_train_HR/0035.png",
    "DIV2K_train_HR/0036.png",
    "DIV2K_train_HR/0037.png",
    "DIV2K_train_HR/0038.png",
    "DIV2K_train_HR/0039.png",
    "DIV2K_train_HR/0040.png",
    "DIV2K_train_HR/0041.png",
    "DIV2K_train_HR/0042.png",
    "DIV2K_train_HR/0043.png",
    "DIV2K_train_HR/0044.png",
    "DIV2K_train_HR/0045.png",
    "DIV2K_train_HR/0046.png",
    "DIV2K_train_HR/0047.png",
    "DIV2K_train_HR/0048.png",
    "DIV2K_train_HR/0049.png",
    "DIV2K_train_HR/0050.png",
    "DIV2K_train_HR/0051.png",
    "DIV2K_train_HR/0052.png",
    "DIV2K_train_HR/0053.png",
    "DIV2K_train_HR/0054.png",
    "DIV2K_train_HR/0055.png",

]

imgoutdir = 'testdata/input/noisy_' + str(dim)
gtoutdir = 'testdata/ground_truth/noisy_' +str(dim)
print('Testing data')

partitionTestset(allimgs,imgoutdir,gtoutdir,nreps,dim,False)