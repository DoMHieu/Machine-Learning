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

'''WARNING: THE TRAINING TAKE HOURS AS THERE ARE 34K VECTORS FOR EACH IMAGE IN 5k IMAGES, SO IF YOU HAVE TIME YOU CAN TRY IT, BUT NOT ADVISED'''
# Get dataset, just put dataset in the same file director
DATASET_DIR = "dataset" 

# Sreach for folder with these name, or close too
CATEGORIES = ['None', 'Smoke', 'Fire'] 

# Resizing image to 64 so the model not gonna crash on it own
IMG_SIZE = 64 

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
    ''' Try to extract'''
    # Calculate histogram for B, G, R channels
    # 3 channels, 32 bins per channel (lowering bins reduces feature size)
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    
    # Normalize the histogram so image brightness doesn't break the model
    cv2.normalize(hist, hist)
    return hist.flatten()

def create_training_data():
    data = []
    labels = []
    
    print(f"Loading images from: {DATASET_DIR}")
    
    for category in CATEGORIES:
        path = os.path.join(DATASET_DIR, category)
        class_num = CATEGORIES.index(category)
        
        if not os.path.exists(path):
            print(f"Warning: Folder '{path}' not found. Skipping.")
            continue
            
        count = 0
        for img_name in os.listdir(path):
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
                
                #COMBINE COLOR AND TEXUTRE FEATURE
                combined_features = np.hstack((color_feats, hog_feats))
                
                data.append(combined_features)
                labels.append(class_num)
                count += 1
                
            except Exception as e:
                pass
        
        print(f"Loaded {count} images for class: {category}")

    return np.array(data), np.array(labels)

def visualize_pca(X, y):
    print("Generating PCA Plot...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
# Ploting to visualize PCA
    plt.figure(figsize=(10, 8))
    colors = ['green', 'gray', 'red'] # Match the order we put the dataset in
    target_names = ['Normal', 'Smoke', 'Fire']
    
    for i, color, target_name in zip([0, 1, 2], colors, target_names):
        # Select indices where label equals i
        indices = np.where(y == i)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], 
                    color=color, alpha=0.6, label=target_name, edgecolors='k')

    plt.title('PCA Visualization of Fire/Smoke Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig('pca_visualization.png')
    print("PCA plot saved as 'pca_visualization.png'")
    plt.show()

# MAIN FLOW THAT RUN THE TRANINING
if __name__ == "__main__":
    
# LOAD DATA
    start_time = time.time()
    X, y = create_training_data()
    print(f"Data loading took: {time.time() - start_time:.2f} seconds")
    print(f"Total images: {len(X)}")
    print(f"Feature vector length per image: {len(X[0])}")

# SPLIT DATA, 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# A standardScaler so that some feature not overpowering other feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# TRAIN
    print("Training SVM... (THIS GONNA TAKE HOURS, YOU HAVE BEEN WARN)")
    clf = svm.SVC(kernel='rbf', C=10, probability=True) 
    clf.fit(X_train_scaled, y_train)
    print("Training complete.")

# Evalutate
    predictions = clf.predict(X_test_scaled)
    print("\nModel Performance")
    print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=CATEGORIES))

# SAVE THE MODEL
    print("Saving model to 'detection_model.pkl'...")
    joblib.dump(clf, 'detection_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Done!")
    print("\nSo, how long did it take")