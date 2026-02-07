import cv2
import joblib
import numpy as np

'''IF MANAGED TO TRAIN IT RAW, TEST TO SEE IT ACCURACY AND HOW MUCH BETTER OR WORSE IT IS COMPARE TO OPTIMIZED ONE'''

print("Loading model...")
model = joblib.load('detection_model.pkl')
scaler = joblib.load('scaler.pkl')


CATEGORIES = ['None', 'Smoke', 'Fire']
IMG_SIZE = 64

def get_hog_features(image):
    # SAme HOG as Training
    hog = cv2.HOGDescriptor((IMG_SIZE,IMG_SIZE), (16,16), (8,8), (8,8), 9)
    return hog.compute(image).flatten()

def get_color_features(image):
    # Same color feature as training
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

'''THE PREDICT FUNCTION'''
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    '''PREPROCESSING (DO THE SAME AS TRAINING)'''
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    

    color_feats = get_color_features(img_resized)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feats = get_hog_features(gray)
    
    features = np.hstack((color_feats, hog_feats))
    features_scaled = scaler.transform([features])
    
    # PREDICT WITH PROBABILITY
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get the highest probability  
    prediction_index = np.argmax(probabilities)
    final_label = CATEGORIES[prediction_index]
    final_confidence = probabilities[prediction_index] * 100

    print(f"\nAnalysis for: {image_path}")
    print("-" * 30)
    
# Check each category to show how much of percent it match the other one
    for i, category in enumerate(CATEGORIES):
        confidence = probabilities[i] * 100
        print(f"{category}: {confidence:.2f}%")
    
    print("-" * 30)
    print(f"RESULT: {final_label} ({final_confidence:.2f}% confident)")
    
# Display image for it more fancy
    text = f"{final_label}: {final_confidence:.1f}%"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_image("test.jpg")