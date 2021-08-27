

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct

# Generate non sparse signal: artificial sound wave
N = 5000
t = np.linspace(0, 1/8, N)

f1 = 200
f2 = 3900
X = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

plt.figure(figsize=[10,4])
plt.plot(t,X)
plt.title('Sound signal')
plt.xlabel('Time (s)')
plt.ylabel('X(t)')

# Compressible representation of non sparse signal
Xdct = dct(X,norm='ortho')
plt.figure(figsize=[10,4])
plt.plot(Xdct)
plt.title('DCT Transform of sound signal')
plt.xlabel('Index')
plt.ylabel('Coefficients')

# Sensing matrix and compression
M=500
Phi= np.random.randn(M, N) / np.sqrt(M)
print ("Compression ratio {0}".format(M/N))  # M/N=0.1
reshape_signal=np.reshape(Xdct, (Xdct.shape[0],1))
y=np.dot(Phi,reshape_signal)

# Reconstruction with Lasso
from sklearn.linear_model import Lasso 

lasso = Lasso(alpha=0.01)
lasso.fit(Phi,y)

plt.figure(figsize=[10,4])
plt.plot(lasso.coef_)
plt.xlabel('Index')
plt.ylabel('Coefficients')
plt.title('Lasso reconstruction')

Xhat = idct(lasso.coef_,norm='ortho')
plt.figure(figsize=[10,4])
plt.plot(t,Xhat)
plt.title('Reconstructed signal')
plt.xlabel('Time (s)')
plt.ylabel('X(t)')

# Relative error
err=np.linalg.norm(X-Xhat,1)/np.linalg.norm(X,1)
print('Reconstruction error is', err)

err=np.linalg.norm(Xdct-lasso.coef_,1)/np.linalg.norm(Xdct,1)
print('Reconstruction error is', err)

# Comparison between an original section and the reconstruction
N=50
t1 = np.linspace(0, 1/8, N)
fig,(ax1, ax2) = plt.subplots(1,2)
ax1.plot(t1,X[0:N])
ax1.set_title('Original section')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('x(t)')
ax2.plot(t1,Xhat[0:N])
ax2.set_title('Recovered section')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('x(t)')
fig.tight_layout(pad=3.0)

