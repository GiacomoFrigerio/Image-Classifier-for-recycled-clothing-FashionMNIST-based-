import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

x_train, y_train = load_reshape("train.npz")
x_test, y_test = load_reshape("test.npz")

#we include a validation set
x_train, x_valid = x_train_[5000:], x_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]


model = tf.keras.Sequential()

# The shape in the first layer must be defined explicitly, other ones will be inferred
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
#padding chooses the correct stride size autonomously
#kernel size is chosen once and keras automatically knows with 1 input he has to do a nxn matrix
#activation is the activation function
#input is (28,28,1) but the output will be (28,28,64) where the last value is the number of filters
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#Downsamples the input along its spatial dimensions (height and width) by
#taking the maximum value over an input window (of size defined by pool_size) for each channel of the input.
#The window is shifted by strides along each dimension.
model.add(tf.keras.layers.Dropout(0.3))
#The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
# 0.3 are erased (set to 0) RANDOMLY
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #I always have to choose this number equal to my number of classes,
#he will give me back the probability of each element to be in that class
#Dense implements the operation: output = activation(dot(input, kernel) + bias)
#where activation is the element-wise activation function passed as the activation argument

model.summary()

#Use categorical cross-entropy (CE) as the loss function and accuracy as the result metrics.
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])



batch_size = 256
epochs = 50

model.fit(x_train_prep, y_train,batch_size=batch_size,epochs=epochs, validation_data=(x_valid_prep, y_valid))

#we load the saved weights with best accuracy saved on disk
model.load_weights('model.weights.best.hdf5')

# We evaluate the model on test set
score = model.evaluate(x_test_prep, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])



