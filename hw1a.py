import glob
import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg

from PIL import Image
from scipy import misc

import theano
import theano.tensor as T
#

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num):
    '''
    This function reconstructs an image X_recon_img given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
        

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]
    
    #TODO: Enter code below for reconstructing the image X_recon_img
    sz = 256/n_blocks

    Img = np.zeros((256,256), dtype=np.float32)

    E = np.dot(D_im,c_im) # (pixel # in a block) x (block # in a image)

    for i in range(0,E.shape[1],1): 
        im = E[:,i]
        im = im.reshape(sz,sz)
        im = np.add(im,X_mean)
    
        Img[int(i/n_blocks)*sz:int(i/n_blocks+1)*sz, int(i%n_blocks)*sz:int(i%n_blocks+1)*sz] = im

    X_recon_img = Img

    #......................
    #......................
    #X_recon_img = ........
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num), cmap=cm.Greys_r)
            
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)
    
    
    
def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    #TODO: Obtain top 16 components of D and plot them

#    print D.shape
#    print sz

    im = np.zeros((sz*sz,1), dtype=np.float32)

    f, axarr = plt.subplots(4,4)
    for i in range(4): #vertical
        for j in range(4): #horizontal
            im = D[:,i*4+j]
            im = im.reshape(sz,sz)
            plt.axes(axarr[i,j])
            plt.imshow(im, cmap=cm.Greys_r)
            
    f.savefig(imname)
    plt.close(f)

    return

    raise NotImplementedError

    
def main():
    '''
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    
    IMAGE_AMOUNT = 200

    Ims = np.zeros((IMAGE_AMOUNT,256,256), dtype=np.float32)

    for dirPath, dirNames, fileNames in os.walk("./Fei_256/"):
#        print dirPath
        
        fileNames.sort()
        idx = 0
        for f in fileNames:
           Ims[idx] = misc.imread(dirPath+f, flatten=True)
           idx += 1

#    print Ims[0] == misc.imread("./Fei_256/image0.jpg", flatten=True)
#    print Ims[2] == misc.imread("./Fei_256/image100.jpg", flatten=True)

    #TODO: Read all images into a numpy array of size (no_images, height, width)

    szs = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        Num_b = 256/sz #Number of blocks in one dim
        Num_block_in_image = (256/sz)**2
        X = np.zeros((IMAGE_AMOUNT*Num_block_in_image, sz*sz))
        
        for i in range(0,IMAGE_AMOUNT,1):
            for j in range(0, Num_block_in_image,1):
                X[i*Num_block_in_image+j] = Ims[i][int(j/Num_b)*sz:int(j/Num_b+1)*sz, int(j%Num_b)*sz:int(j%Num_b+1)*sz].flatten()

#        print Ims[0]
#        print X[0]
#        print X[31]
        

        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        ''' 
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        
        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)

        X_CM = np.dot(X.transpose(), X) # covariance matrix
        eigenVal, eigenVec = linalg.eigh(X_CM)

        idx = eigenVal.argsort()[::-1]   
        eigenVal = eigenVal[idx]
        eigenVec = eigenVec[:,idx] 
       
        D = eigenVec

#        print eigenVal
#        print eigenVec

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        
        c = np.dot(D.T, X.T)
        
        for i in range(0, IMAGE_AMOUNT, 10):
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((sz, sz)), n_blocks=int(256/sz), im_num=i)

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
    
    
