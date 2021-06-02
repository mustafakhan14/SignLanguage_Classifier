import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
import warnings
from sklearn.metrics import confusion_matrix
sns.set()
warnings.filterwarnings("ignore")

df_train, df_test = pd.read_csv('../data/sign_mnist_train.csv'), pd.read_csv('../data/sign_mnist_test.csv')
y_train, y_test = df_train['label'], df_test['label']
df_train.drop(['label'], axis=1, inplace=True)
df_test.drop(['label'], axis=1, inplace=True)

input_size, channels, batch_size, num_epochs = 28, 1, 128, 100
x_train = df_train.values.reshape(df_train.shape[0], input_size, input_size, channels)
x_test = df_test.values.reshape(df_test.shape[0], input_size, input_size, channels)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

dg = ImageDataGenerator(rescale=1./255, zoom_range=0.2, width_shift_range=.2, height_shift_range=.2, rotation_range=30, brightness_range=[0.8, 1.2], horizontal_flip=True)
dg_scaled = ImageDataGenerator(rescale=1./255)
x_train, x_test = dg.flow(x_train, y_train, batch_size=batch_size), dg_scaled.flow(x_test, y_test)
callback_checkpoint = ModelCheckpoint(filepath='best_model.hdf5', save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True)
callback_lr = ReduceLROnPlateau(monitor='loss', mode='min', min_delta=0.01, patience=3, factor=.75, min_lr=0.00001, verbose=1)

Model = Sequential([Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(input_size, input_size, channels)),
                    MaxPool2D(2, 2, padding='same'), Dropout(0.2),
                    Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
                    MaxPool2D(2, 2, padding='same'), Dropout(0.2),
                    Conv2D(filters=512, kernel_size=(3, 3), activation="relu"),
                    MaxPool2D(2, 2, padding='same'), Dropout(0.2),
                    Flatten(),
                    Dense(units=4096, activation="relu"), Dropout(0.2),
                    Dense(units=1024, activation="relu"), Dropout(0.2),
                    Dense(units=256, activation="relu"), Dropout(0.2),
                    Dense(units=25, activation="softmax"),
                    ])

Model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = Model.fit(x_train, validation_data=x_test, epochs=num_epochs, callbacks=[callback_checkpoint, callback_lr])
accuracy, val_accuracy = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs_trained = range(len(accuracy))

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_trained, accuracy, label='Training Accuracy')
plt.plot(epochs_trained, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_trained, loss, label='Training Loss')
plt.plot(epochs_trained, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('Accuracy, Loss.png')

score = Model.evaluate(x_test)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
df_test = pd.read_csv('../data/sign_mnist_test.csv')
y_test, x_test = df_test['label'], df_test.values.reshape(df_test.shape[0], input_size, input_size, channels)
df_test.drop(['label'], axis=1, inplace=True)
y_pred = np.argmax(Model.predict(x_test), axis=1)

"""Generate confusion matrix."""
plt.figure(figsize=(15, 15))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
plt.savefig('confusion.png')