

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# Estimation

N = 1000
mu = 1 
sigma = 1
x = np.random.normal(mu, sigma, N)
l = (1/N)*np.ones(N)

normx= np.linalg.norm(x)
print('Norm x', normx)
norml= np.linalg.norm(l)
print('Norm l', norml)

# Inner product of l and x
InProd = l@x
print('Inner product', InProd)

Mean_error_OE = np.zeros(9)
Mean_error_DE = np.zeros(9)
values = [100,200,300,400,500,600,700,800,900]
iterations = np.ones(300)
niter = len(iterations)

for j in [0,1,2,3,4,5,6,7,8]:
    M = values[j]
    sum_error_OE =0
    sum_error_DE =0
    
    for i in iterations:
        Phi = np.random.randn(M,N)/ np.sqrt(M)
        OE=(N/M)*x.T@Phi.T@np.linalg.inv(Phi@Phi.T)@Phi@l
        Error_OE= abs(OE-InProd)/(norml*normx)
        sum_error_OE = sum_error_OE + Error_OE
        DE=(Phi@l)@(Phi@x)
        Error_DE= abs(DE-InProd)/(norml*normx)
        sum_error_DE = sum_error_DE + Error_DE
        

        
    Mean_error_OE[j]=sum_error_OE/niter
    Mean_error_DE[j] =sum_error_DE/niter


r = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plt.plot(r,Mean_error_OE,'o',label="Orthogonalized Estimator")
plt.plot(r,Mean_error_DE,'o',label="Direct Estimator")
plt.legend()
plt.xlabel('M/N')
plt.ylabel('Estimation error')
plt.title('Estimation error vs. M/N')
plt.show()

