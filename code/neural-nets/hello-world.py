# Link: Deep Learning with Python, TensorFlow, and Keras tutorial (https://www.youtube.com/watch?v=wQ8BIBpya2k)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 28x28 of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data() # unpacking
x_train = tf.keras.utils.normalize(x_train, axis=1) # normalizing data (aka scaling it)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# building model
model = tf.keras.models.Sequential() # feedforward model
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # first layer - input layer; we flatten the data
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # second layer - hidden layer; adds 128 units, default activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # third layer - hidden layer; adds 128 units, default activation function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # fourth layer - output layer; adds 10 outputs, softmax activation function

# training parameters
model.compile(optimizer='adam', # default optimizer is adam
            loss='sparse_categorical_crossentropy', # calculates loss using sparse categorical crossentropy (this minimizes the degree of error)
            metrics=['accuracy']) # metrics we want to use

# run the model
model.fit(x_train, y_train, epochs=3)

# calculating the validation loss and validation accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print('Validation Loss: ' + str(val_loss) + '\n' + 'Validation Accuracy: ' + str(val_acc))

# saving model
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

# predicts model (QUICK THING: We added input_shape=(28,28) to when we use Flatten to make model.save() work)
predictions = new_model.predict([x_test])
print(predictions) # prints 1-hot arrays aka probability distributions

# shows the 0th index of the dataset (first prints it, then displays it's graphic)
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()

''' 
# Shows Basic Data
print(x_train[0])

# Shows Data Image
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
'''