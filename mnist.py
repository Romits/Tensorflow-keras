import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
MIN_EPOCHS = 25

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch >= MIN_EPOCHS and logs.get('acc') > 0.99):
            print("\n Reached accuracy of 99% on epoch {} so stopping training".format(epoch))
            self.model.stop_training = True

def main():
    mnist = tf.keras.datasets.mnist
    callbacks = myCallback()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, callbacks=[callbacks])
    test_loss = model.evaluate(x_test, y_test)

    print("Accuracy on the test data is {}".format(test_loss[1] *100))


if __name__=="__main__":
    main()
