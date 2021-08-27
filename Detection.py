

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# Detection
SNR = 100 # Decibelios
alpha = np.linspace(0,1,50)

def Q(x): # Q(x) = 1-P(N(0,1)<=x)
    #result = 1- norm.cdf(x)
    result= norm.cdf(-x)
    return result

def Q_inv(x): 
    #result= norm.ppf(1-x)
    result = -norm.ppf(x)
    return result

def P_alpha(alpha,r,SNR): # r= Compression ratio M/N
    result= Q(Q_inv(alpha)-math.sqrt(r*SNR))
    return result

plt.plot(alpha, P_alpha(alpha,0.05,SNR),label="r=0.05")
plt.plot(alpha, P_alpha(alpha,0.1,SNR), label = "r=0.1")
plt.plot(alpha, P_alpha(alpha,0.2,SNR), label = "r=0.2")
plt.plot(alpha, P_alpha(alpha,0.4,SNR), label = "r=0.4")
plt.legend()
plt.xlabel('False alarm rate')
plt.ylabel('Detection rate')
plt.title('Detection rate vs. False alarm rate')
plt.show()


alpha = 0.1 
r = np.linspace(0,1,100)
def Q(x): 
    result= norm.cdf(-x)
    return result

def Q_inv(x): 
    result = -norm.ppf(x)
    return result

def P_alpha(alpha,r,SNR): 
    result= Q(Q_inv(alpha)-np.sqrt(r*SNR))
    return result

plt.plot(r, P_alpha(alpha,r,316.23),label="SNR=25 dB")
plt.plot(r, P_alpha(alpha,r,100), label = "SNR=20 dB")
plt.plot(r, P_alpha(alpha,r,31.62), label = "SNR=15 dB")
plt.plot(r, P_alpha(alpha,r,10), label = "SNR=10 dB")
plt.legend()
plt.xlabel('M/N')
plt.ylabel('Detection rate')
plt.title('Detection rate vs. M/N')
plt.show()

