from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten

def CNN_training():
    
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape =(256,256,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    return model

def save_model(model):
    
    model.save('cnn_model.h5')
    
    