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


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,im_num):
    '''
    This function reconstructs an image given the number of
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
    '''
    
    c_im = c[:num_coeffs,im_num]
    D_im = D[:,:num_coeffs]
    
    #TODO: Enter code below for reconstructing the image
    
    Img = np.zeros((256,256), dtype=np.float32)

    E = np.dot(D_im,c_im)

    Img = E.reshape(256,256)
    Img = np.add(Img,X_mean)

    X_recon_img = Img

    #......................
    #......................
    #X_recon_img = ........
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number_of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,im_num), cmap=cm.Greys_r)
            
    f.savefig('output/hw1b_{0}.png'.format(im_num))
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
    Read here all images(grayscale) from Fei_256 folder and collapse 
    each image to get an numpy array Ims with size (no_images, height*width).
    Make sure the images are read after sorting the filenames
    '''

    IMAGE_AMOUNT = 200
    DATA_SIZE = 65536

    Ims = np.zeros((IMAGE_AMOUNT,DATA_SIZE), dtype=np.float32)

    for dirPath, dirNames, fileNames in os.walk("./Fei_256/"):
        fileNames.sort()
        idx = 0
        for f in fileNames:
           im = misc.imread(dirPath+f, flatten=True)
           Ims[idx] = np.reshape(im, DATA_SIZE)
           idx += 1


    #TODO: Write a code snippet that performs as indicated in the above comment
    
    Ims = Ims.astype(np.float32)
    X_mean = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mean.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''

    COM_AMOUNT = 16
    
    D = np.zeros((DATA_SIZE,COM_AMOUNT), dtype=np.float32)
    Lamda = np.zeros(COM_AMOUNT, dtype=np.float32)    

    init_d_i = np.zeros(DATA_SIZE, dtype=np.float32)
    init_d_i[0] = 1.0
    d_i = theano.shared(init_d_i, name="d_i")

    #Theano function
    x_in = T.fmatrix('x_in')
    d_in = T.fmatrix('d_in')
    l_in = T.fvector('l_in')
    grad = -2*T.dot(x_in.T,T.dot(x_in,d_i)) + np.sum(2*l_in[j]*T.dot(d_in[:,j],T.dot((d_in[:,j]).T,d_i)) for j in xrange(COM_AMOUNT))

    #debug
#    grad_t = np.sum(2*Lamda[j]*T.dot(d_in[:,j],T.dot((d_in[:,j]).T,d_i)) for j in xrange(COM_AMOUNT))
    grad_t   = l_in[0]*T.dot(d_in[:,0],T.dot((d_in[:,0]).T,d_i))
    grad_t_1 = 2*Lamda[1]*T.dot(d_in[:,1],T.dot((d_in[:,1]).T,d_i))

    step = 0.001
    y_i = d_i - step*grad
    Y_i = y_i / T.dot(y_i.T,y_i)**0.5

#    print d_i.type()
#    print Y_i.astype(theano.config.floatX).type()


    train = theano.function(inputs=[x_in, d_in, l_in], outputs=[Y_i], updates=[(d_i, Y_i.astype(theano.config.floatX))])    

    #debug
    show_grad   = theano.function([l_in, d_in],[grad_t])
    show_grad_1 = theano.function([d_in],[grad_t_1])


    for i in range(0,COM_AMOUNT,1):
        cnt = 0
        GO = True
        d_i.set_value(init_d_i)

#        print ("i:", i)
        while(cnt<100 and GO):
            d_i_last = d_i.get_value()
            
            #debug
#            print ("0: " ,show_grad(Lamda,D))
#            print ("1: " ,show_grad_1(D))
#            print ("D[:,0]: ", D[:,0])
#            print ("D[:,1]: ", D[:,1])


            train(X,D,Lamda)

#            print ("d_i_last: ", d_i_last)
#            print ("     d_i: ", d_i.get_value())

            if (np.square(d_i_last-d_i.get_value())).sum() < 1.0e-14:
                GO = False
            cnt += 1

#        print grad.eval()
#        print ( np.sum(2*Lamda[j]*T.dot(D[:,j],T.dot(np.transpose(D[:,j]),d_i)) for j in range(COM_AMOUNT))).eval()

        D[:,i] = d_i.get_value()
        Lamda[i] = T.dot(T.dot(X,d_i).T,T.dot(X,d_i)).eval()  

#        print ("i: ", i)
#        print ("D[:,0]: ", D[:,0])
#        print ("D[:,1]: ", D[:,1])

#        print (i, ":", grad.eval())


#    print (np.square(D[:,0])).sum()





    '''
    print Lamda[0]
    print D[:,0]
    print Lamda[1]
    print D[:,1]
    print Lamda[2]
    print D[:,2]
    print Lamda[3]
    print D[:,3]
    print Lamda[4]
    print D[:,4]
    print Lamda[5]
    print D[:,5]
    print Lamda[6]
    print D[:,6]
    print Lamda[7]
    print D[:,7]
    print Lamda[8]
    print D[:,8]
    print Lamda[9]
    print D[:,9]
    print Lamda[10]
    print D[:,10]
    print Lamda[11]
    print D[:,11]
    print Lamda[12]
    print D[:,12]
    print Lamda[13]
    print D[:,13]
    print Lamda[14]
    print D[:,14]
    print Lamda[15]
    print D[:,15]
    '''

    c = np.dot(D.T, X.T)


    
    #TODO: Write a code snippet that performs as indicated in the above comment
        
    for i in range(0, IMAGE_AMOUNT, 10):
        plot_reconstructions(D=D, c=c, num_coeff_array=[1, 2, 4, 6, 8, 10, 12, 14, 16], X_mean=X_mean.reshape((256, 256)), im_num=i)

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')


if __name__ == '__main__':
    main()
    
    
