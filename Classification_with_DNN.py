

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

def apply_projection(image,phi,N,nx,ny):
    resized_img = np.reshape(image, (N, 1))
    y = phi@resized_img
    result=phi.transpose()@y
    resized_result = np.reshape(result, (nx, ny))
    return resized_result

def preprocess_dataset(images):
    sample = images[1]
    nx,ny=sample.shape
    N = nx*ny
    m = round(N*0.5)
    A = np.random.randn(N,N) 
    phi = A[0:m,:] / np.sqrt(m)
    
    
    new_images=np.array([apply_projection(image,phi,N,nx,ny) for image in images])
    return new_images
    
train_x = preprocess_dataset(train_x)
test_x = preprocess_dataset(test_x)

train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)

val_x = train_x[:5000]
val_y = train_y[:5000]

model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=20, kernel_size=(5, 5), activation='relu', input_shape=train_x[0].shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(keras.layers.Conv2D(filters=50, kernel_size=(5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units=500))

model.add(keras.layers.Dense(units=10, activation = 'softmax'))

optimizer='adam'
model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10, validation_data=(val_x, val_y))

scores=model.evaluate(test_x, test_y)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(scores)

# A few random samples
import random
list = range(10000)
num=10
use_samples = random.sample(list, num)
samples_to_predict=[]

num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    sample=use_samples[i]
    ax = axes[i//num_col, i%num_col]
    ax.imshow(test_x[sample], cmap='gray')
    samples_to_predict.append(test_x[sample])
    ax.set_title('Label: {}'.format(test_y[sample]))
plt.tight_layout()
plt.show()

# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)
print(samples_to_predict.shape)

# Generate predictions for samples
predictions = model.predict(samples_to_predict)

# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print(classes)

r=[0.1,0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
acc_g = [0.51,0.74, 0.80, 0.87, 0.90, 0.91, 0.94, 0.95, 0.93,0.93]
plt.plot(r,acc_g,'o')
plt.xlabel('M/N')
plt.ylabel('Accuracy')
plt.title('Classification of MNIST')

