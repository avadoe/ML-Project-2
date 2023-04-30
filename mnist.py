import tensorflow
import numpy as np
import keras
from keras.datasets import mnist
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = 10
batch_size = 64
num_epochs = 10

architectures = [   {'num_hidden_layers': 2, 'hidden_layer_size': 0},    
                    {'num_hidden_layers': 2, 'hidden_layer_size': 0},    
                    {'num_hidden_layers': 2, 'hidden_layer_size': 0},    
                    {'num_hidden_layers': 3, 'hidden_layer_size': 0},    
                    {'num_hidden_layers': 3, 'hidden_layer_size': 0},   
                    {'num_hidden_layers': 3, 'hidden_layer_size': 0},
                ]

activation_functions = ['tanh', 'relu', 'sigmoid']
output_functions = ['tanh', 'sigmoid', 'softmax']

results = []
kf = KFold(n_splits=10, shuffle=True)



for train_index, test_index in kf.split(X_train):
    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]
    for i, architecture in enumerate(architectures):
        if architecture['num_hidden_layers'] == 2:
            num_neurons = [50, 50] if random.random() == 0.5 else [30, 70] if random.random() < 0.5 else [70, 30]
        else:
            num_neurons = [25, 25, 50] if random.random() == 0.5 else [25, 50, 25] if random.random() < 0.5 else [50, 25, 25]
            
        architecture['hidden_layer_size'] = num_neurons
        for activation in activation_functions:
            for output_fn in output_functions:
                model = Sequential()
                model.add(Conv2D(filters=32, kernel_size=(3, 3), activation=activation, input_shape=(28, 28, 1)))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Flatten())
                for i in range(architecture['num_hidden_layers']):
                    model.add(Dense(units=num_neurons[i], activation=activation))
                model.add(Dense(units=num_classes, activation=output_fn))               
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
                model.fit(X_cv_train, y_cv_train, batch_size=batch_size, epochs=num_epochs, verbose=0)
                y_predict = np.argmax(model.predict(X_cv_test), axis=1)
                accuracy = accuracy_score(np.argmax(y_cv_test, axis=1), y_pred=y_predict)
                cm = confusion_matrix(np.argmax(y_cv_test, axis=1), y_pred=y_predict)
                results.append({'architecture': architecture, 'activation function': activation, 'output function': output_fn, 
                                'accuracy': accuracy, 'confusion matrix': cm})
            
for i, result in enumerate(results):
    print(f'Model {i+1}: Architecture={result["architecture"]}, Activation Function={result["activation function"]}, Output Function={result["output function"]},Accuracy={result["accuracy"]:.4f}')
    print(result['confusion matrix'])