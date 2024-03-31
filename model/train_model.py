import tensorflow as tf
from model import build_model
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model():
    df = pd.read_csv('data/preprocessed_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model()
    
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
                tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)]
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=callback)

if __name__ == "__main__":
    train_model()