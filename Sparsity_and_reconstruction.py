

import numpy as np
import matplotlib.pyplot as plt

#Generate s-Sparse 1D signal of length N
N=1000 #Length
S=10 #Sparsity
x = np.zeros((N,1))
x[0:S,:] = np.random.randn(S,1)
np.random.shuffle(x)
plt.plot(x)
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Sparse signal')
plt.show()

# Random sensing matrix
# 200 measurements taken
M = 200
Phi = np.random.randn(M,N) / np.sqrt(M)
print ("Compression ratio {0}".format(M/N)) # M/N=0.2

y = Phi @x # Measurement vector


def omp(A,y,S):

    x_k = np.zeros_like(x)
    r_k = np.copy(y)
    Omega_k = []
    
    for k in range(S):
        lambda_k = np.argmax(np.abs(A.T @ r_k))
        Omega_k.append(lambda_k)
        x_k[Omega_k,:] = np.linalg.pinv(A[:,Omega_k]) @ y
        r_k = y - A @ x_k
        
    return x_k


omp_recovery = omp(Phi,y,S)
error= np.linalg.norm(x-omp_recovery,1)/np.linalg.norm(x,1)

print('Reconstruction error is', error)

fig,(ax1, ax2) = plt.subplots(1,2)
ax1.plot(omp_recovery)
ax1.set_title('OMP Reconstructed')
ax1.set_ylabel('Amplitude')
ax1.set_xlabel('Index')
    
ax2.plot(x)
ax2.set_title('Original Signal')
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Index')
fig.tight_layout(pad=3.0)
plt.show()

# Hard-thresholding operator
def hard_thresholding(x, k):
    x = x.flatten()
    length = x.size
    H_k_x = np.zeros_like(x)
    indices = list(np.argsort(np.absolute(x))[::-1][0:k].flatten())
    H_k_x[indices] = x[indices]
    H_k_x = np.reshape(H_k_x, (length, 1))
    
    return indices, H_k_x
    
# Reconstruction with CoSamp
def cosamp(A,y,S):

    x_previous = np.zeros_like(x)
    r = np.copy(y)
    T = []
    A_pinv_T_k = np.zeros_like(A)
    
    for k in range(S):
        Omega = hard_thresholding(A.T @ r,2*S)[0]
        supp_x_previous = x_previous.flatten().nonzero()[0].tolist()
        T = list(set().union(T,Omega,supp_x_previous))
        A_pinv_T_k[:,T] = A[:,T]
        x_k = hard_thresholding(np.linalg.pinv(A_pinv_T_k) @ y, S)[1]
        
        r = y - A @ x_k
        
    return x_k
cosamp_recovery = cosamp(Phi,y,S)
error_cosamp = np.linalg.norm(x-cosamp_recovery,1)/np.linalg.norm(x,1)
print('Reconstruction error is', error_cosamp)

fig,(ax1, ax2) = plt.subplots(1,2)
ax1.plot(cosamp_recovery)
ax1.set_title('CoSaMP Reconstructed')
ax1.set_ylabel('Amplitude')
ax1.set_xlabel('Index')
    
ax2.plot( x)
ax2.set_title('Original Signal')
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Index')
fig.tight_layout(pad=3.0)
plt.show()

