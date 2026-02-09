import numpy as np
import joblib
import os
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

if __name__ == "__main__":
    if not os.path.exists("data_train_X.npy"):
        print("Error: data_train_X.npy not found.")
        exit()

    with tqdm(total=4, desc="Training Pipeline", unit="step") as pbar:
        pbar.set_description("Step 1/4: Loading Data")
        try:
            X_train = np.load("data_train_X.npy")
            y_train = np.load("data_train_y.npy")
            
            X_test = None
            if os.path.exists("data_test_X.npy"):
                X_test = np.load("data_test_X.npy")
                y_test = np.load("data_test_y.npy")
            
            tqdm.write(f" -> Loaded {len(X_train)} training samples.")
            pbar.update(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            exit()

        pbar.set_description("Step 2/4: Training Model (This takes time...)")
        clf = SVC(kernel='rbf', C=1000, gamma='scale', probability=True, cache_size=2000)
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        
        tqdm.write(f" -> Training finished in {end_time - start_time:.2f} seconds.")
        pbar.update(1)
        pbar.set_description("Step 3/4: Evaluating")
        if X_test is not None:
            predictions = clf.predict(X_test)
            if hasattr(predictions, 'to_numpy'): predictions = predictions.to_numpy()
            predictions = predictions.astype(int)
            y_test = y_test.astype(int)
            acc = accuracy_score(y_test, predictions)
            tqdm.write(f" -> Accuracy: {acc*100:.2f}%")
            tqdm.write("-" * 30)
            tqdm.write(classification_report(y_test, predictions))
            tqdm.write("-" * 30)
        else:
            tqdm.write(" -> No test data found, skipping evaluation.")
            
        pbar.update(1)
        pbar.set_description("Step 4/4: Saving Model")
        joblib.dump(clf, "model.pkl")
        model_name = "new_model_svm_cuml.pkl"
        try:
            joblib.dump(clf, model_name)
            tqdm.write(f" -> Success! Model saved as {model_name}")
        except PermissionError:
            tqdm.write(" -> Permission denied. Saving to /tmp/ instead...")
            backup_path = "/tmp/model_svm_cuml.pkl"
            joblib.dump(clf, backup_path)
            tqdm.write(f" -> Model saved to: {backup_path}")
        except Exception as e:
            tqdm.write(f" -> Error saving model: {e}")
        pbar.update(1)

    print("\n✅ Process Complete!")