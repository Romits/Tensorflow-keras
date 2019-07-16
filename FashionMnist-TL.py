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


#Resize and convert greyscale to RGB
def transform_image(imageDataset):
    imageDataset = array([cv2.resize(image, (32,32))for image in imageDataset])
    imageDataset = array([cv2.cvtColor(image,cv2.COLOR_GRAY2RGB) for image in imageDataset])
    return imageDataset


def main():


   vgg19_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))

   #print(vgg19_model.summary())

   for layer in vgg19_model.layers:
       layer.trainable = False

   last_layer = vgg19_model.get_layer('block5_pool')
   print("shape of last layer:", last_layer.output_shape)
   last_output = last_layer.output


   fmnist = tensorflow.keras.datasets.fashion_mnist
   callbacks = myCallback()
   (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

   training_images = transform_image(training_images)
   test_images = transform_image(test_images)

   training_images = training_images.reshape(60000, 32, 32, 3)
   test_images = test_images.reshape(10000, 32, 32, 3)
   images_val, images_test, labels_val, labels_test = train_test_split(test_images, test_labels, test_size=0.20)


   # Use the image generator

   train_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

   validation_datagen = ImageDataGenerator(rescale=1./255)

   train_generator = train_datagen.flow(x=training_images, y=training_labels, batch_size=12, shuffle=True )
   validation_generator = validation_datagen.flow(x=images_val, y=labels_val, batch_size=12, shuffle=True)

   x = tensorflow.keras.layers.GlobalAveragePooling2D()(last_output)
   #x = tensorflow.keras.layers.Dense(128, activation='relu')(x)

   #x = tensorflow.keras.layers.Dropout(0.2)(x)
   x = tensorflow.keras.layers.Dense(10, activation='softmax')(x)

   model = tensorflow.keras.Model(vgg19_model.input, x)
   #print(model.summary())
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   history= model.fit_generator(train_generator,
                                steps_per_epoch=5000,
                                epochs=20,
                                validation_data=validation_generator,
                                validation_steps=833,
                                verbose=2,callbacks = [callbacks])
   #print(model.summary())
   #test_loss, test_acc = model.evaluate(test_images, test_labels)
   print("Accuracy for the model is {}".format(history.history.get('acc')[-1] * 100))
   print("Validation accuracy for the model is {}".format(history.history.get('val_acc')[-1] * 100))

   test_loss, test_acc = model.evaluate(images_test, labels_test)
   print("Test accuracy for the model is {}".format(test_acc *100))

if __name__=='__main__':
    main()
