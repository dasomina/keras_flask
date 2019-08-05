import tensorflow as tf
import keras

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
Y_1hot = enc.fit_transform(Y.reshape(-1, 1)).toarray()

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y_1hot, epochs=300, batch_size=10)
print('Model training is complete')

file_name = 'iris_model.h5'
model.save(file_name) # model을 저장한다
print('Model Saved in ' + file_name)
