import numpy as np
import joblib
import os
import time
from sklearn.svm import SVC
from tqdm import tqdm
import model_evaluation

if __name__ == "__main__":
    TRAIN_X_PATH = "data_train_X.npy"
    TRAIN_Y_PATH = "data_train_y.npy"
    MODEL_OUTPUT_NAME = "new_model_svm_cuml.pkl"

    if not os.path.exists(TRAIN_X_PATH):
        print(f"❌ Error: '{TRAIN_X_PATH}' not found. Please run preprocessing first.")
        exit()

    print("🚀 STARTING TRAINING PIPELINE...")

    with tqdm(total=3, desc="Pipeline", unit="step") as pbar:
        
        pbar.set_description("Step 1/3: Loading Data")
        try:
            X_train = np.load(TRAIN_X_PATH)
            y_train = np.load(TRAIN_Y_PATH)
            tqdm.write(f" -> Loaded {len(X_train)} training samples.")
            pbar.update(1)
        except Exception as e:
            print(f"❌ Data Load Error: {e}")
            exit()

        pbar.set_description("Step 2/3: Training Model (Please wait...)")
        clf = SVC(kernel='rbf', C=1000, gamma='scale', probability=True, cache_size=2000)
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        
        tqdm.write(f" -> Training finished in {end_time - start_time:.2f} seconds.")
        pbar.update(1)

        pbar.set_description("Step 3/3: Saving Model")
        try:
            joblib.dump(clf, MODEL_OUTPUT_NAME)
            joblib.dump(clf, "model.pkl") 
            tqdm.write(f" -> Model saved to: {MODEL_OUTPUT_NAME}")
        except Exception as e:
            tqdm.write(f"❌ Save Error: {e}")
        pbar.update(1)

    print("\n✅ TRAINING COMPLETE!")

    model_evaluation.evaluate_model(model_path=MODEL_OUTPUT_NAME)