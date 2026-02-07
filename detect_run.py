import os
import cv2
import numpy as np
import joblib
from tkinter import *
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
from skimage.feature import local_binary_pattern

# --- CONFIGURATION ---
CROP_SIZE = 128
GRID_SIZE = 4
STRIDE = 32  # Smaller = more accurate but slower (e.g., 16 or 24)

OBJECT_NAMES = {
    0: "Background",
    1: "Object_A", # Replace with your real name
    2: "Object_B"  # Replace with your real name
}

MODEL_FILES = {
    "scaler": "model_scaler.pkl",
    "pca": "model_pca.pkl",
    "svm": "model.pkl" 
}

# --- HELPER: Merges overlapping boxes ---
def nms(boxes, overlapThresh=0.3):
    if len(boxes) == 0: return []
    boxes = np.array(boxes).astype("float")
    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return pick

class SlidingWindowDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Monday make me broken")
        self.root.geometry("800x850")
        self.root.configure(bg="#2c3e50")

        self.models_loaded = False
        try:
            self.scaler = joblib.load(MODEL_FILES["scaler"])
            self.pca = joblib.load(MODEL_FILES["pca"])
            self.clf = joblib.load(MODEL_FILES["svm"])
            self.models_loaded = True
            self.status_msg, self.status_color = "Models Loaded. Drop Image.", "#2ecc71"
        except Exception as e:
            self.status_msg, self.status_color = f"Load Error: {str(e)[:30]}", "#e74c3c"

        self.setup_ui()

    def setup_ui(self):
        self.lbl_info = Label(self.root, text=self.status_msg, fg=self.status_color, bg="#2c3e50", font=("Arial", 12, "bold"))
        self.lbl_info.pack(pady=10)

        self.canvas = Canvas(self.root, width=700, height=550, bg="#ecf0f1", highlightthickness=2)
        self.canvas.pack(pady=10)
        self.canvas.create_text(350, 275, text="DROP PHOTO TO SCAN", font=("Arial", 14), fill="#7f8c8d")

        self.res_var = StringVar(value="Result: Waiting...")
        Label(self.root, textvariable=self.res_var, fg="#f1c40f", bg="#2c3e50", font=("Arial", 14, "bold")).pack(pady=10)

        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind('<<Drop>>', self.on_image_drop)

    def extract_features(self, patch):
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        c_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(c_hist, c_hist)
        
        gray = hsv[:,:,2]
        h, w = gray.shape
        h_s, w_s = h // GRID_SIZE, w // GRID_SIZE
        l_feat = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = gray[i*h_s:(i+1)*h_s, j*w_s:(j+1)*w_s]
                lbp = local_binary_pattern(cell, 24, 3, method="uniform")
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
                l_feat.extend(hist.astype("float") / (hist.sum() + 1e-7))
        return np.hstack((c_hist.flatten(), l_feat)).reshape(1, -1)

    def on_image_drop(self, event):
        if not self.models_loaded: return
        path = event.data.strip('{}')
        if not os.path.isfile(path): return

        img = cv2.imread(path)
        # Resize large images to keep scanning fast
        if max(img.shape) > 800:
            scale = 800 / max(img.shape)
            img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

        self.res_var.set("Scanning image... please wait")
        self.root.update()

        detections, labels = [], []
        # --- SLIDING WINDOW SCAN ---
        for y in range(0, img.shape[0] - CROP_SIZE, STRIDE):
            for x in range(0, img.shape[1] - CROP_SIZE, STRIDE):
                patch = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
                feat = self.scaler.transform(self.extract_features(patch))
                pca_f = self.pca.transform(feat)
                pred = int(self.clf.predict(pca_f)[0])

                if pred != 0: # If not background
                    detections.append([x, y, x + CROP_SIZE, y + CROP_SIZE])
                    labels.append(pred)

        draw_img = img.copy()
        if detections:
            indices = nms(detections)
            for i in indices:
                x1, y1, x2, y2 = detections[i]
                name = OBJECT_NAMES.get(labels[i], f"ID:{labels[i]}")
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw_img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.res_var.set(f"Complete")
        else:
            self.res_var.set("No objects found")

        # Display result
        rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert("RGB")
        pil_img.thumbnail((700, 550))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(350, 275, image=self.tk_img)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = SlidingWindowDetector(root)
    root.mainloop()