

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


# Cancel then recover with noise for M=200
N=1000
M=200
ks=10
kI=20


# Supports
index= np.random.choice(N, ks+kI, replace=False)
index_xI=np.random.choice(index,kI, replace=False)
index_xs = np.array(list(e for e in index if e not in index_xI))

# Signal of interest and interference
xI = np.zeros((N,1))
xI[index_xI] =np.random.randn(kI,1)
xs = np.zeros((N,1))
xs[index_xs] =np.random.randn(ks,1)
xI= xI*np.sqrt(np.sum(xs**2)/np.sum(xI**2))
x=xs+xI

plt.plot(x)
plt.title('x=xs+xI')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.show()

# Added noise
MSE= np.sum(x**2)/N 
sigma = np.sqrt(MSE/10**(1.5))
noise = np.random.normal(0, sigma, (N,1))
plt.plot(noise)
plt.title('Noise')
plt.show()

norm_square_x = np.sum(x**2)
norm_square_xI = np.sum(xI**2)
norm_square_xs = np.sum(xs**2)
norm_square_noise = np.sum(noise**2)
print("Norm_square_x = %s" %norm_square_x)
print("Norm_square_xI = %s" %norm_square_xI)
print("Norm_square_xs = %s" %norm_square_xs)
print("Norm_square_noise = %s" %norm_square_noise)

test =x+ noise
plt.plot(test)
plt.title('Test signal: x+noise')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.show()

# Measurements
Phi = np.random.randn(M,N)/np.sqrt(M)
y=Phi@test

# Projector onto Omega orthogonal
PsiI = np.zeros((N, kI))
for column , index in enumerate ( index_xI ):
    PsiI[index,column]=1

#PsiI=np.eye(N,kI)
Omega = Phi@PsiI
Omega_dagg=np.linalg.inv(Omega.T@Omega)@Omega.T
P = np.eye(M,M)-Omega@Omega_dagg
Py=P@y
s = np.count_nonzero(Py)
plt.plot(Py)
plt.title('Projection of y onto subspace orthogonal to the column space of Omega')
plt.show()

#Reconstruction with CoSamp
def hard_thresholding(x, k):
    x = x.flatten()
    length = x.size
    H_k_x = np.zeros_like(x)
    indices = list(np.argsort(np.absolute(x))[::-1][0:k].flatten())
    H_k_x[indices] = x[indices]
    H_k_x = np.reshape(H_k_x, (length, 1))
    
    return indices, H_k_x


def cosamp(A,Py,S):

    x_previous = np.zeros_like(xs)
    r = np.copy(Py)
    T = []
    A_pinv_T_k = np.zeros_like(A)
            
    for k in range(S):
        Omega = hard_thresholding(A.T @ r,2*S)[0]
        supp_x_previous = x_previous.flatten().nonzero()[0].tolist()
        T = list(set().union(T,Omega,supp_x_previous))
        A_pinv_T_k[:,T] = A[:,T]
        x_k = hard_thresholding(np.linalg.pinv(A_pinv_T_k) @ Py, S)[1]
        r = Py - A @ x_k
            
    return x_k
        
        
A = P@Phi
cosamp_recovery = cosamp(A,Py,ks)


error_cosamp = np.linalg.norm(xs-cosamp_recovery,1)/np.linalg.norm(xs,1)
print('Reconstruction error is', error_cosamp)

fig,(ax1, ax2) = plt.subplots(1,2)
ax1.plot(cosamp_recovery)
ax1.set_title('CoSAMP Reconstructed')
ax1.set_xlabel('Index')
ax1.set_ylabel('Amplitude')
    
ax2.plot(xs)
ax2.set_title('Original Signal xs')
ax2.set_xlabel('Index')
ax2.set_ylabel('Amplitude')
fig.tight_layout(pad=3.0)
plt.show()

#SNR of the reconstruction
numerator = np.sum(xs**2)/N
denominator = np.sum((xs-cosamp_recovery)**2)/N 
square = math.sqrt(numerator/denominator)
SNR= 20*math.log10(square)
print("SNR = %s" % (SNR))




# Cancel then recover with noise
N=1000
ks=10
values=[10, 20, 30, 40, 50, 60, 70, 80, 90]
SNR=np.zeros(9)
iterations = np.ones(50)
niter = len(iterations)

for i in [0,1,2,3,4,5,6,7,8]:
    kI=values[i]
    sum_SNR=0
    for j in iterations:
        # Supports
        index= np.random.choice(N, ks+kI, replace=False)
        index_xI=np.random.choice(index,kI, replace=False)
        index_xs = np.array(list(e for e in index if e not in index_xI))
        
        # Signal of interest and interference
        xI = np.zeros((N,1))
        xI[index_xI] =np.random.randn(kI,1)
        xs = np.zeros((N,1))
        xs[index_xs] =np.random.randn(ks,1)
        xI= xI*np.sqrt(np.sum(xs**2)/np.sum(xI**2))
        x=xs+xI
        
        # Added noise
        MS= np.sum(x**2)/N # Mean Square 
        sigma = np.sqrt(MS/10**(1.5))
        noise = np.random.normal(0, sigma, (N,1))
        test =x+ noise
        
        # Measurements
        Phi = np.random.randn(M,N)/np.sqrt(M)
        y=Phi@test
        
        # Projector onto Omega orthogonal
        PsiI = np.zeros((N, kI))
        for column , index in enumerate ( index_xI ):
            PsiI[index,column]=1
            
        Omega = Phi@PsiI
        Omega_dagg=np.linalg.inv(Omega.T@Omega)@Omega.T
        P = np.eye(M,M)-Omega@Omega_dagg
        Py=P@y
        
        #Reconstruction with CoSamp
        def hard_thresholding(x, k):
            x = x.flatten()
            length = x.size
            H_k_x = np.zeros_like(x)
            indices = list(np.argsort(np.absolute(x))[::-1][0:k].flatten())
            H_k_x[indices] = x[indices]
            H_k_x = np.reshape(H_k_x, (length, 1))
            return indices, H_k_x
        
        def cosamp(A,Py,S):

            x_previous = np.zeros_like(xs)
            r = np.copy(Py)
            T = []
            A_pinv_T_k = np.zeros_like(A)
            
            for k in range(S):
                Omega = hard_thresholding(A.T @ r,2*S)[0]
                supp_x_previous = x_previous.flatten().nonzero()[0].tolist()
                T = list(set().union(T,Omega,supp_x_previous))
                A_pinv_T_k[:,T] = A[:,T]
                x_k = hard_thresholding(np.linalg.pinv(A_pinv_T_k) @ Py, S)[1]
                r = Py - A @ x_k
            
            return x_k
        
        
        A = P@Phi
        cosamp_recovery = cosamp(A,Py,ks)
        
        
        #SNR of the reconstruction
        numerator = np.sum(xs**2)/N
        denominator = np.sum((xs-cosamp_recovery)**2)/N 
        square = math.sqrt(numerator/denominator)
        sum_SNR= sum_SNR + 20*math.log10(square)
    
    SNR[i]=sum_SNR/niter

r = [1, 2, 3, 4, 5, 6, 7, 8, 9] # kI/ks   
plt.plot(r, SNR,'o')
plt.title('Cancel then recover')
plt.xlabel('kI/ks')
plt.ylabel('SNR')
plt.show()

