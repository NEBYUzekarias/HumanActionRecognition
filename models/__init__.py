from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,MaxPool2D,LSTM,Bidirectional,TimeDistributed,Flatten

def get_human_action_recognition_model(input_shape,num_class,max_sequence_length=30):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32,(3,3),(1,1),padding='valid',activation="relu",input_shape=(max_sequence_length,input_shape[0],input_shape[1],input_shape[2]))))
    model.add(TimeDistributed(Conv2D(64,(3,3),(1,1,),"valid",activation="relu")))
    model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(128,(3,3),(1,1,),"valid",activation="relu")))
    model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(1024,activation="relu")))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(Bidirectional(LSTM(64,return_sequences=True,stateful=False,activation='tanh')))
    model.add(Bidirectional(LSTM(128,return_sequences=False,stateful=False,activation='tanh')))
    model.add(Dense(1024,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_class,activation="softmax"))

    return model
