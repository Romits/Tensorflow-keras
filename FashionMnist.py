import tensorflow
import wget
from numpy import array
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import applications


class myCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.99):
            print("\n Reached accuracy of 99% on epoch {} so stopping training".format(epoch))
            self.model.stop_training = True



def main():

   fmnist = tensorflow.keras.datasets.fashion_mnist
   callbacks = myCallback()
   (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

   # Need to reshape the data as CNN needs data in the format M*H*W*C

   training_images = training_images.reshape(60000, 28, 28, 1)
   test_images = test_images.reshape(10000, 28, 28, 1)
   images_val, images_test, labels_val, labels_test = train_test_split(test_images, test_labels, test_size=0.20)

   train_datagen = ImageDataGenerator(rescale=1./255)
   validation_datagen = ImageDataGenerator(rescale=1. / 255)


   train_generator = train_datagen.flow(x=training_images, y=training_labels, batch_size=16, shuffle=True )
   validation_generator = validation_datagen.flow(x=images_val, y=labels_val, batch_size=16, shuffle=True)


   model = Sequential()
   model.add(Conv2D(64,(2,2), activation='relu', input_shape=(28,28,1)))
   model.add(MaxPooling2D(2,2))
   model.add(Dropout(0.3))
   model.add(Conv2D(32,(2,2), activation='relu'))
   model.add(MaxPooling2D(2,2))
   model.add(Dropout(0.3))
   model.add(Flatten())
   model.add(Dense(256, activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(10, activation='softmax'))
   print(model.summary())
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   history= model.fit_generator(train_generator,
                                steps_per_epoch=5000,
                                epochs=2,
                                validation_data=validation_generator,
                                validation_steps=833,
                                verbose=2,callbacks = [callbacks])
   print("Accuracy for the model is {}".format(history.history.get('acc')[-1] * 100))
   print("Validation accuracy for the model is {}".format(history.history.get('val_acc')[-1] * 100))

   test_loss, test_acc = model.evaluate(images_test, labels_test)
   print("Test accuracy for the model is {}".format(test_acc *100))

if __name__=='__main__':
    main()
