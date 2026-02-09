import numpy as np
import joblib
import os
import sys
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path="new_model_svm_cuml.pkl", x_path="data_test_X.npy", y_path="data_test_y.npy"):
    print("\n" + "="*40)
    print("📊 MODEL EVALUATION")
    print("="*40)

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at '{model_path}'")
        return
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"⚠️ Warning: Test data not found ('{x_path}'). Skipping evaluation.")
        return

    try:
        print(f" -> Loading model: {model_path}...")
        clf = joblib.load(model_path)
        
        print(f" -> Loading test data...")
        X_test = np.load(x_path)
        y_test = np.load(y_path)
        print(f" -> Test samples: {len(X_test)}")

        print(" -> Running predictions...")
        predictions = clf.predict(X_test)
        
        if hasattr(predictions, 'to_numpy'): 
            predictions = predictions.to_numpy()
        predictions = predictions.astype(int)
        y_test = y_test.astype(int)

        acc = accuracy_score(y_test, predictions)
        print("\n" + "-"*30)
        print(f"🏆 ACCURACY: {acc*100:.2f}%")
        print("-"*30)
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(y_test, predictions))
        print("-"*30)
        
    except Exception as e:
        print(f"❌ Evaluation Error: {e}")

if __name__ == "__main__":
    evaluate_model()