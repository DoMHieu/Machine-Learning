import cv2
import joblib
import numpy as np


IMG_SIZE = 64
CATEGORIES = ['None', 'Smoke', 'Fire']

print("Loading models...")
model = joblib.load('detection_model_optimized.pkl')
scaler = joblib.load('scaler_optimized.pkl')
pca = joblib.load('pca.pkl')

def get_hog_features(image):
    win_size = (IMG_SIZE, IMG_SIZE)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog.compute(image).flatten()

def get_color_features(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def predict_image(image_path):
    print(f"\nAnalyzing: {image_path}")
    

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    color_feats = get_color_features(img_resized)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feats = get_hog_features(gray)
    
    features = np.hstack((color_feats, hog_feats))
    
    features_scaled = scaler.transform([features])
    
    # PCA Transform
    # This compresses the features to match what the SVM was trained on
    features_pca = pca.transform(features_scaled)
    
    probabilities = model.predict_proba(features_pca)[0]
    prediction_index = np.argmax(probabilities)
    
    final_label = CATEGORIES[prediction_index]
    final_confidence = probabilities[prediction_index] * 100
    
    print("-" * 30)
    for i, category in enumerate(CATEGORIES):
        print(f"{category}: {probabilities[i]*100:.2f}%")
    print("-" * 30)
    print(f"RESULT: {final_label} ({final_confidence:.2f}%)")

    cv2.putText(img, f"{final_label} ({final_confidence:.1f}%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_image("test_fire.jpg")