

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from matplotlib import pyplot as plt

(x_train_o, y_train), (x_test_o, y_test) = mnist.load_data() # Load MNIST database
x_train_o=x_train_o.astype(float)
x_test_o=x_test_o.astype(float)

x_train = np.reshape(x_train_o, (60000, 784)) # Transform images of 28x28 into arrays of 28*28
x_train, y_train = shuffle(x_train, y_train) # Shuffle the images of the training set

ns = 500 # We take ns images for training: half of them are of digit 3 and the other half do not contain it
l_tr=np.where(y_train==3)
l2_tr=np.where(y_train!=3)
x_train_d=np.array(x_train)
y_train_d=np.array(y_train)
print(y_train_d.shape)
print(x_train_d.shape)
x_train[0:ns//2][:] = x_train_d[l_tr[0][0:ns//2]][:]
y_train[0:ns//2]= y_train_d[l_tr[0][0:ns//2]]
x_train[ns//2:ns][:] = x_train_d[l2_tr[0][0:ns//2]][:]
y_train[ns//2:ns]= y_train_d[l2_tr[0][0:ns//2]][:]
print(np.sum(y_train==3))

y_train=y_train[0:ns] # We keep just those ns images selected for training
x_train=x_train[0:ns][:]

y_train[0:ns//2]=1 # Images with digit 3 have label 1
y_train[ns//2:ns]=0 # Images without digit 3 have label 0

x_train, y_train = shuffle(x_train, y_train) # Shuffle the images selected

x_test = np.reshape(x_test_o, (10000, 784)) # As before, we select nt images for testing and re-label them. Half os them are of digit 3
nt = 2000
l_tr=np.where(y_test==3)
l2_tr=np.where(y_test!=3)
x_test_d=np.array(x_test)
y_test_d=np.array(y_test)
x_test[0:nt//2][:] = x_test_d[l_tr[0][0:nt//2]][:]
y_test[0:nt//2]=1
x_test[nt//2:nt][:] = x_test_d[l2_tr[0][0:nt//2]][:]
y_test[nt//2:nt]=0
x_test = x_test[0:nt][:]
y_test = y_test[0:nt]

x_test, y_test = shuffle(x_test, y_test) # Shuffle images selected

# Loop to obtain a graphic of accuracy vs. m (number of measurements taken)
n = 784
values = list(range(780,100,-60))
l=len(values)
m_values=list(range(0,l))

test_acc_g=np.zeros(l)
test_acc_b=np.zeros(l)

niter =20
iterations = np.ones(niter)

for j in m_values:
    m=values[j]
    print(m)
    acc_g_sum=0
    acc_b_sum=0
    for i in iterations:
        A = np.random.randn(m,n)
        B=np.random.binomial(1,0.5,(m,n))
        B[np.where(B==0)]=-1
        x_train_g = np.matmul(x_train, A.T)/m**0.5
        x_test_g = np.matmul(x_test, A.T)/m**0.5
        x_train_b = np.matmul(x_train, B.T)/m**0.5
        x_test_b = np.matmul(x_test, B.T)/m**0.5

        kfold = StratifiedKFold(n_splits=10)

        svclassifier = SVC()
        svc_param_grid = {'kernel': ['linear'],
                          'C': [0.01, 0.1, 1, 10, 50, 100, 200, 300, 1000]}
        gsSVM_g = GridSearchCV(svclassifier,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
        gsSVM_g.fit(x_train_g, y_train)

        SVM_best_g = gsSVM_g.best_estimator_
  

        gsSVM_b = GridSearchCV(svclassifier,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
        gsSVM_b.fit(x_train_b, y_train)

        SVM_best_b = gsSVM_b.best_estimator_
  
        
        acc_g_sum= acc_g_sum+np.sum(SVM_best_g.predict(x_test_g) == y_test)/nt
        
        acc_b_sum= acc_b_sum+ np.sum(SVM_best_b.predict(x_test_b) == y_test)/nt
        
    test_acc_g[j]=acc_g_sum/niter
    test_acc_b[j]=acc_b_sum/niter
    
r = [x / n for x in values]
plt.plot(r,test_acc_g,'o',label='Gaussian')
plt.plot(r,test_acc_b,'o',label='Bernoulli')
plt.xlabel('M/N')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title('Detection of digit 3')
plt.show()

