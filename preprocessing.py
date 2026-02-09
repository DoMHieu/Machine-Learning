import os
import cv2
import numpy as np
import glob
import joblib
import random
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA_ROOT = "data"
CROP_SIZE = 128
GRID_SIZE = 4
BATCH_SIZE = 1000
N_COMPONENTS = 512

def get_spatial_lbp_features(image):
    radius = 3
    n_points = 8 * radius
    h, w = image.shape
    h_step = h // GRID_SIZE
    w_step = w // GRID_SIZE
    features = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell = image[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            lbp = local_binary_pattern(cell, n_points, radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            features.extend(hist)
    return np.array(features)

def get_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return interArea / float(boxAArea + 1e-5)

def parse_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path): return []
    with open(label_path, 'r') as f: lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - w/2) * img_w)
        y1 = int((cy - h/2) * img_h)
        x2 = int((cx + w/2) * img_w)
        y2 = int((cy + h/2) * img_h)
        my_cls = 2 if cls == 1 else 1
        boxes.append([my_cls, x1, y1, x2, y2])
    return boxes

def process_folder_smart_crop(folder_name):
    full_path = os.path.join(DATA_ROOT, folder_name)
    img_dir = os.path.join(full_path, "images")
    lbl_dir = os.path.join(full_path, "labels")
    image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    X_list = []
    y_list = []
    for img_path in tqdm(image_paths, desc=f"Processing {folder_name}", unit="img"):
        try:
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            txt = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            boxes = parse_yolo_boxes(os.path.join(lbl_dir, txt), w, h)
            
            for box in boxes:
                cls, bx1, by1, bx2, by2 = box
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 > bx1 and by2 > by1:
                    roi = img[by1:by2, bx1:bx2]
                    roi = cv2.resize(roi, (CROP_SIZE, CROP_SIZE))
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:,:,2]
                    l_feat = get_spatial_lbp_features(gray)
                    c_feat = get_color_features(roi)
                    X_list.append(np.hstack((c_feat, l_feat)))
                    y_list.append(cls)

            num_neg = 1 if len(boxes) > 0 else 2
            for _ in range(num_neg):
                rw = random.randint(50, min(w, 400))
                rh = random.randint(50, min(h, 400))
                rx = random.randint(0, w - rw)
                ry = random.randint(0, h - rh)
                candidate = [rx, ry, rx+rw, ry+rh]
                is_clean = True
                for box in boxes:
                    if iou(candidate, box[1:]) > 0.1:
                        is_clean = False; break
                if is_clean:
                    roi = img[ry:ry+rh, rx:rx+rw]
                    roi = cv2.resize(roi, (CROP_SIZE, CROP_SIZE))
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:,:,2]
                    l_feat = get_spatial_lbp_features(gray)
                    c_feat = get_color_features(roi)
                    X_list.append(np.hstack((c_feat, l_feat)))
                    y_list.append(0)
        except: pass
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

if __name__ == "__main__":
    print("\n--- START PREPROCESSING ---")
    X_train, y_train = process_folder_smart_crop("train")
    print(f" -> Collected {len(X_train)} training samples.")
    X_test, y_test = process_folder_smart_crop("test")
    print(f" -> Collected {len(X_test)} testing samples.")
    
    if len(X_train) == 0: 
        print("Error: No training data found!")
        exit()
    
    print(f"\nFeature Extraction Complete. Raw Dimension: {X_train.shape[1]}")
    print("Fitting Scaler and PCA (this may take a moment)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=N_COMPONENTS)
    X_train_final = pca.fit_transform(X_train_scaled)
    
    X_train_final = X_train_final.astype(np.float32)
    np.save("data_train_X.npy", X_train_final)
    np.save("data_train_y.npy", y_train)
    
    if X_test is not None and len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
        X_test_final = pca.transform(X_test_scaled)
        X_test_final = X_test_final.astype(np.float32)
        np.save("data_test_X.npy", X_test_final)
        np.save("data_test_y.npy", y_test)

    joblib.dump(scaler, "model_scaler.pkl")
    joblib.dump(pca, "model_pca.pkl")
    print("\n✅ Done Preprocessing. Models and data saved.")