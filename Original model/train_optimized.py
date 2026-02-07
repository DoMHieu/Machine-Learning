import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Get dataset, just put dataset in the same file director
DATASET_DIR = "dataset" 

# Search for folder with these name, or close too
CATEGORIES = ['None', 'Smoke', 'Fire'] 

# Resizing image to 64 so the model not gonna crash on it own
IMG_SIZE = 64 

# OPTIMIZATION CONFIG
CACHE_FILE_X = "cached_features_X.npy"
CACHE_FILE_Y = "cached_labels_y.npy"
USE_PCA = True
PCA_VARIANCE = 0.95 # REDUCE NOISE

def get_hog_features(image):
    # In simple term, it turn images into vector to train
    win_size = (IMG_SIZE, IMG_SIZE)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute vector for it to be more "flatten"
    hog_feats = hog.compute(image).flatten()
    return hog_feats

def get_color_features(image):
    ''' Try to extract feature'''
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    
    # Normalize so the birghtness dont break the model
    cv2.normalize(hist, hist)
    return hist.flatten()

def create_training_data():
#CHECK CACHE FROM OPTIMIZATION
    if os.path.exists(CACHE_FILE_X) and os.path.exists(CACHE_FILE_Y):
        print(f"Found cached data! Loading from {CACHE_FILE_X}...")
        try:
            X = np.load(CACHE_FILE_X)
            y = np.load(CACHE_FILE_Y)
            print(f"Loaded {len(X)} images from cache.")
            return X, y
        except Exception as e:
            print("Cache not found, reloading images...")

    data = []
    labels = []
    
    print(f"Loading images from: {DATASET_DIR}")
    
    for category in CATEGORIES:
        path = os.path.join(DATASET_DIR, category)
        try:
            class_num = CATEGORIES.index(category)
        except ValueError:
            continue 
        
        if not os.path.exists(path):
            print(f"Warning: Folder '{path}' not found. Skipping.")
            continue
            
        count = 0
        files = os.listdir(path)
        
        for i, img_name in enumerate(files):
            try:
                # READ IMAGES
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                
                if img_array is None:
                    continue
                
                # RESIZE
                img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                # GET THE COLOR FEATURE
                color_feats = get_color_features(img_resized)
                
                # GET TEXTURE FEATURE
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                hog_feats = get_hog_features(gray)
                
                # COMBINE COLOR AND TEXTURE FEATURE
                combined_features = np.hstack((color_feats, hog_feats))
                
                data.append(combined_features)
                labels.append(class_num)
                count += 1
                
                # Print progess to know where it at
                if count % 100 == 0:
                    print(f"Processed {count} images for {category}...")
                
            except Exception as e:
                pass
        
        print(f"Loaded {count} images for class: {category}")

    X = np.array(data)
    y = np.array(labels)

# OPTIMIZATION
    print("Saving features to cache files...")
    np.save(CACHE_FILE_X, X)
    np.save(CACHE_FILE_Y, y)

    return X, y

# MAIN FLOW THAT RUN THE TRAINING
if __name__ == "__main__":
    
    start_time = time.time()
    X, y = create_training_data()
    print(f"Data loading took: {time.time() - start_time:.2f} seconds")
    print(f"Total images: {len(X)}")
    print(f"Original Feature vector length: {len(X[0])}")

#SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SCALER
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- OPTIMIZATION: TRAIN WITH PCA ---
    if USE_PCA:
        print(f"Applying PCA (Keeping {PCA_VARIANCE*100}% variance)...")
        pca = PCA(n_components=PCA_VARIANCE)
        
        # Fit PCA on Training data
        X_train_final = pca.fit_transform(X_train_scaled)
        # Transform Test data the same
        X_test_final = pca.transform(X_test_scaled)
        
        print(f"Reduced Feature vector length: {X_train_final.shape[1]}")
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
        pca = None

    # TRAIN
    print("Training SVM... (this might take a few minutes)")
    # CACHE FOR OPTIMIZATION
    clf = svm.SVC(kernel='rbf', C=10, probability=True, cache_size=1000) 
    clf.fit(X_train_final, y_train)

    # EVALUATE
    predictions = clf.predict(X_test_final)
    print("\nModel Performance")
    print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=CATEGORIES))

    # SAVE THE MODEL
    print("Saving model to 'detection_model_optimized.pkl'...")
    joblib.dump(clf, 'detection_model_optimized.pkl')
    joblib.dump(scaler, 'scaler_optimized.pkl')
    
    if pca is not None:
        joblib.dump(pca, 'pca.pkl')
        print("Saved PCA model to 'pca.pkl' ")
        
    print("Done!")