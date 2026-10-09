from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras        import regularizers

def nn_sequencial(input_size=128):
    model = Sequential()
    
    model.add(Input(shape=(input_size, input_size, 3)))
        
    model.add(Conv2D(16, (3,3), strides=1, padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.5))  
    
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.0001))) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.5))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.7))  

    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.7))  
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])

    return model