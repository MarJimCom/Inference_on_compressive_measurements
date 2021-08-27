

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import math
from skimage import io
from pylbfgs import owlqn

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    
def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things: 
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[ri].reshape(b.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

# fractions of the scaled image to randomly sample at
sample_sizes = (0.1, 0.01)

# read original image
Xorig = io.imread("leon.PNG")
ny,nx,nchan = Xorig.shape

# for each sample size
Z = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
masks = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
for i,s in enumerate(sample_sizes):

    # create random sampling index vector
    k = round(nx * ny * s)
    ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

    # for each color channel
    for j in range(nchan):

        # extract channel
        X = Xorig[:,:,j].squeeze()

        # create images of mask (for visualization)
        Xm = 255 * np.ones(X.shape)
        Xm.T.flat[ri] = X.T.flat[ri]
        masks[i][:,:,j] = Xm

        # take random samples of image, store them in a vector b
        b = X.T.flat[ri].astype(float)

        # perform the L1 minimization in memory
        Xat2 = owlqn(nx*ny, evaluate, None, 5)

        # transform the output back into the spatial domain
        Xat = Xat2.reshape(nx, ny).T # stack columns
        Xa = idct2(Xat)
        Z[i][:,:,j] = Xa.astype('uint8')
        
relative_error1 = np.linalg.norm((Xorig-Z[0]).flat,1)/np.linalg.norm((Xorig).flat,1)
relative_error2 = np.linalg.norm((Xorig-Z[1]).flat,1)/np.linalg.norm((Xorig).flat,1)
print('Reconstruction error  for 10% sample is', relative_error1)
print('Reconstruction error  for 1% sample is', relative_error2)
fig,(ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(Xorig)
ax1.set_title('Original image')
ax1.axis('off')
ax2.imshow(Z[0])
ax2.set_title('10% sample')
ax2.axis('off')
ax3.imshow(Z[1])
ax3.set_title('1% sample')
ax3.axis('off')
plt.show()


N=nx*ny*nchan
numerator = np.sum(Xorig**2)/N
denominator = np.sum((Xorig-Z[0])**2)/N 
square = math.sqrt(numerator/denominator)
SNR= 20*math.log10(square)
print("SNR for sample_size = 0.1 is %s" % (SNR))

denominator2 = np.sum((Xorig-Z[1])**2)/N #MSE
square2 = math.sqrt(numerator/denominator2)
SNR= 20*math.log10(square2)
#print(SNR)
print("SNR for sample_size = 0.01 is %s" % (SNR))

