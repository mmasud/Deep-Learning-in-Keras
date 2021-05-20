import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop('price_range', axis=1).to_numpy()
    # define min max scaler
    scaler = StandardScaler()
    # transform data
    scaled_X= scaler.fit_transform(X)
    y= df.price_range.to_numpy()
    y= to_categorical(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(scaled_X, y, test_size = 0.2, random_state = 2, stratify=y)
    
    # Model (Keras)
    # Initialization
    num_features= X.shape[1] # 20
    num_classes= 4

    model = Sequential([
        Dense(512, input_dim=num_features, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.summary()

    # Compile
    model.compile(optimizer="adam", loss= "categorical_crossentropy", metrics="accuracy")

    # Fit
    history = model.fit(X_train, y_train,
          batch_size = 64, epochs = 30, verbose = 2,
          validation_data=(X_val, y_val));

    # Evaluate

    score= model.evaluate(X_val, y_val, verbose=0)
    print('Test loss: {}%'.format(score[0] * 100))
    print('Test score: {}%'.format(score[1] * 100))
    print("MLP Error: %.2f%%" % (100 - score[1] * 100))
    
    # Result visualization
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(x = history.epoch, y = history.history['loss'])
    sns.lineplot(x = history.epoch, y = history.history['val_loss'])
    ax.set_title('Learning Curve (Loss)')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'test'], loc='best')
    plt.show()

    # Confusion Matrix
    # To find the predicted class for each test example, need to use axis=1.
    pred= model.predict(X).argmax(axis=1)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(confusion_matrix(y.argmax(axis=1), pred), cmap='Blues', 
                annot=True, cbar=False, fmt='d', square=True, linewidth=0.4, ax=ax)

    ax.set_ylabel('Pred', fontweight='bold')
    ax.set_xlabel('True', fontweight='bold')

    plt.show()