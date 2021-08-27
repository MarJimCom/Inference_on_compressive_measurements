

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# Classification
R=3
N=1000
mu = 0 #mean
sigma = 1 # standard deviation
S1 = np.random.normal(mu, sigma, N)
S2 = np.random.normal(mu, sigma, N)
S3 = np.random.normal(mu, sigma, N)
d1 = np.linalg.norm(S1-S2,2)
d2 = np.linalg.norm(S1-S3,2)
d3 = np.linalg.norm(S2-S3,2)
d = min([d1, d2, d3])

M=0
values = [100,200,300,400,500,600,700,800,900]
Prob_error = np.zeros((9,3))

for j in [0,1,2,3,4,5,6,7,8]:
    M = values[j]
    Phi = np.random.randn(M,N)/ np.sqrt(M)
    SNR=[10,15,20] #dB
    for k in [0,1,2]:
        sigma_noise =d/(10**(SNR[k]/20))
        niter=300
        iterations = np.ones(niter)
        sum_error = np.zeros(3)
        
        for i in iterations:
            n = np.random.normal(0, sigma_noise, N)
            x=S1+n
            y=Phi@x
            t1=(y-Phi@S1).T@np.linalg.inv(Phi@Phi.T)@(y-Phi@S1)
            t2=(y-Phi@S2).T@np.linalg.inv(Phi@Phi.T)@(y-Phi@S2)
            t3=(y-Phi@S3).T@np.linalg.inv(Phi@Phi.T)@(y-Phi@S3)
            minimum = min([t1,t2,t3])
            if minimum==t1:
                sum_error[k] = sum_error[k]
            else :
                sum_error[k] = sum_error[k] + 1
        
        Prob_error[j,k]=sum_error[k]/niter

    
r = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plt.plot(r,Prob_error[:,0],'o',label="SNR = 10 dB")
plt.plot(r,Prob_error[:,1],'o',label="SNR = 15 dB")
plt.plot(r,Prob_error[:,2],'o',label="SNR = 20 dB")
plt.legend()
plt.xlabel('M/N')
plt.ylabel('Probability of error')
plt.title('Classification error vs. M/N')

