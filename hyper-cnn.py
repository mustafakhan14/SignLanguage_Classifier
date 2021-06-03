import tensorflow as tf
from tf.layers import Dense
from tf.models import Sequential
from tf.layers import Dropout
from tf.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

print(tf.VERSION)
print(tf.keras.__version__)

# repeat some of the initial values here so we make sure they were not changed
input_dim = x_train.shape[1]
num_classes = 10


# function that creates the model (required for KerasClassifier) while accepting the hyperparameters to tune
def create_model_2(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_initializer=init, activation=tf.nn.relu))
    model.add(Dense(num_classes, kernel_initializer=init, activation=tf.nn.softmax))

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


# create the sklearn model for the network
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
model_init_batch_epoch_CV = KerasClassifier(build_fn=create_model_2, verbose=1)

# initializers that came at the top in our previous cross-validation chosen
init_mode = ['glorot_uniform', 'uniform']
batches = [128, 256, 512]
epochs = [20, 50, 100]

# grid search for initializer, batch size and number of epochs
param_grid = dict(epochs=epochs, batch_size=batches, init=init_mode)
grid = GridSearchCV(estimator=model_init_batch_epoch_CV, param_grid=param_grid, cv=3)
grid_result = grid.fit(x_train, y_train)

# print results
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')
