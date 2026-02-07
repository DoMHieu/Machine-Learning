import numpy as np
import joblib
import os
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
    if not os.path.exists("data_train_X.npy"): exit()
        
    X_train = np.load("data_train_X.npy")
    y_train = np.load("data_train_y.npy")
    
    X_test = None
    if os.path.exists("data_test_X.npy"):
        X_test = np.load("data_test_X.npy")
        y_test = np.load("data_test_y.npy")
    
    print(f"Training on {len(X_train)} samples...")
    
    clf = SVC(kernel='rbf', C=1000, gamma='scale', probability=True, cache_size=2000)
    clf.fit(X_train, y_train)
    
    print("Training finished.")

    if X_test is not None:
        predictions = clf.predict(X_test)
        if hasattr(predictions, 'to_numpy'): predictions = predictions.to_numpy()
        predictions = predictions.astype(int)
        y_test = y_test.astype(int)
        
        acc = accuracy_score(y_test, predictions)
        print(f"Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, predictions))

    joblib.dump(clf, "model.pkl")
    print("Model saved.")

    model_name = "new_model_svm_cuml.pkl"
    
    try:
        # 2. Try to save locally first
        print(f"Attempting to save {model_name}...")
        joblib.dump(clf, model_name)
        print(f"Success! Model saved as {model_name}")
    except PermissionError:
        # 3. If that fails, save to the Linux 'tmp' folder (which always works)
        print("Permission denied on current folder. Saving to /tmp/ instead...")
        backup_path = "/tmp/model_svm_cuml.pkl"
        joblib.dump(clf, backup_path)
        print(f"Model successfully saved to: {backup_path}")
        print("You can move it later using: cp /tmp/model_svm_cuml.pkl .")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Training process complete.")