import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import pandas as pd
from model import build_model

def evaluate_model():
    # Load the best model
    model = tf.keras.models.load_model('best_model.h5')
    
    # Load test data
    df_test = pd.read_csv('data/test_data.csv')
    X_test = df_test.drop('target', axis=1)
    y_test = df_test['target']
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype("int32")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    print(f"Confusion Matrix:\n{cm}")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # Plotting the ROC curve is left as an exercise to the reader, using matplotlib or similar library

if __name__ == "__main__":
    evaluate_model()
