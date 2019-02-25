from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Dropout

def model():
# Initialising the ANN
 classifier = Sequential()

# Adding the input layer and the first hidden layer
 classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 10))

# Adding the second hidden layer
 classifier.add(Dense(units = 12, activation = 'sigmoid'))

# Third layer
 classifier.add(Dense(units = 10, activation = 'sigmoid'))
 classifier.add(Dropout(0.1))
 classifier.add(Dense(units=10,activation='sigmoid'))
 classifier.add(Dropout(0.1))
 classifier.add(Dense(units=9,activation='sigmoid'))


# Adding the output layer
 classifier.add(Dense(units = 9, activation = 'tanh'))

 classifier.add(Dense(units = 3, activation = 'sigmoid'))

# Compiling the ANN
 classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
 return classifier
