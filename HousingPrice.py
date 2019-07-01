# Since no activation is specified, we will have linear activation
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
MIN_EPOCHS=500
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs={}):
        if(epoch >= MIN_EPOCHS and logs.get('acc') > 0.49):
            print("\n Reached desired accuracy after {} epocs so cancelling training!".format(epoch))
            self.model.stop_training = True
        
    
def main():
    callbacks = myCallback()
    model = Sequential()
    model.add(Dense(1,input_shape=(1,)))
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0], dtype=float)
    ys = np.array([1.0, 1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5], dtype=float)
    model.fit(xs,ys,epochs=1000, callbacks=[callbacks])
    print(model.predict([7.0]))

if __name__=="__main__":
    main()
