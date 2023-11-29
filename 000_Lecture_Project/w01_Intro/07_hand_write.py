# Handwritten digits classification using neural network
# In this notebook we will classify handwritten digits using a simple neural network 
# which has only input and output layers. 
# We will than add a hidden layer and see how the performance of the model improves
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
print('len(X_train):', len(X_train)) # 60000
print('len(X_test):', len(X_test)) # 10000
print('X_train[0].shape:', X_train[0].shape)  # (28, 28)
print('X_train[0]:', X_train[0])  # 
plt.matshow(X_train[0])
plt.show()

print('y_train[0]:', y_train[0])  # 5
X_train = X_train / 255  # We scale to have better accuracy
X_test = X_test / 255

print('X_train[0]:', X_train[0])

X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

print('X_train_flattened.shape:', X_train_flattened.shape)
print('X_train_flattened[0]:', X_train_flattened[0])

# Very simple neural network with no hidden layers¶
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5) # loss: 0.2666 - accuracy: 0.9257
model.evaluate(X_test_flattened, y_test) # [0.267718106508255, 0.9262999892234802]
y_predicted = model.predict(X_test_flattened)
print('y_predicted[0]:', y_predicted[0])

plt.matshow(X_test[0])
plt.show()
print('np.argmax(y_predicted[0]):', np.argmax(y_predicted[0])) # 7
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print('y_predicted_labels:', y_predicted_labels)  # [7, 2, 1, 0, 4]

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print('cm:', cm) # confusion matrix will be plot by seaborn 

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
# Using hidden layer
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
print(model.evaluate(X_test_flattened,y_test)) # [0.08831490576267242, 0.9764000177383423]
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
#  Using Flatten layer so that we don't have to call .reshape on input dataset
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
print(model.evaluate(X_test,y_test))  # [0.08721642196178436, 0.9761000275611877]