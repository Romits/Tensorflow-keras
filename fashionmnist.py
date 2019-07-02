import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
MIN_EPOCHS = 10


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch >= MIN_EPOCHS and logs.get('acc') > 0.98):
            print("\n Reached accuracy of 99% on epoch {} so stopping training".format(epoch))
            self.model.stop_training = True

def main():

   fmnist = tf.keras.datasets.fashion_mnist
   callbacks = myCallback()
   (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
   # Need to reshape the data as CNN needs data in the format M*H*W*C
   training_images = training_images.reshape(60000, 28, 28, 1)
   training_images =training_images/255.0
   test_images = test_images.reshape(10000,28,28,1)
   test_images = test_images/255.0
   model = Sequential()
   model.add(Conv2D(64,(3,3), activation='relu', input_shape=(28,28,1)))
   model.add(MaxPooling2D(2,2))
   model.add(Conv2D(64,(3,3), activation='relu'))
   model.add(MaxPooling2D(2,2))
   model.add(Flatten())
   model.add(Dense(128, activation='relu'))
   model.add(Dense(10, activation='softmax'))
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(training_images, training_labels, epochs=20, callbacks = [callbacks])
   #print(model.summary())
   test_loss, test_acc = model.evaluate(test_images, test_labels)
   print(test_acc * 100)


if __name__=='__main__':
    main()




