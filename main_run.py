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

# --- CUSTOMIZE YOUR LABELS HERE ---
# Change these strings to match what you actually trained your model on
OBJECT_NAMES = {
    0: "Background",
    1: "Smoke",    # Example: Change this to your real object name
    2: "Fire"    # Example: Change this to your real object name
}

MODEL_FILES = {
    "scaler": "model_scaler.pkl",
    "pca": "model_pca.pkl",
    "svm": "model.pkl" 
}

class InstantDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("But a Fork in the Microwave Momento")
        self.root.geometry("600x700")
        self.root.configure(bg="#2c3e50")

        print(f"--- DIAGNOSTIC ---")
        print(f"Current Working Directory: {os.getcwd()}")
        
        self.models_loaded = False
        try:
            self.check_files_exist()
            print("Loading models...")
            self.scaler = joblib.load(MODEL_FILES["scaler"])
            self.pca = joblib.load(MODEL_FILES["pca"])
            self.clf = joblib.load(MODEL_FILES["svm"])
            print("SUCCESS: All models loaded.")
            self.models_loaded = True
            self.status_msg = "System Ready. Drop Image."
            self.status_color = "#2ecc71"
        except Exception as e:
            print(f"ERROR: {e}")
            self.status_msg = f"Error: {str(e)[:30]}"
            self.status_color = "#e74c3c"

        self.setup_ui()

    def check_files_exist(self):
        for name, filename in MODEL_FILES.items():
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Missing '{filename}'")

    def setup_ui(self):
        self.lbl_info = Label(self.root, text=self.status_msg, fg=self.status_color, bg="#2c3e50", font=("Arial", 12, "bold"))
        self.lbl_info.pack(pady=15)

        self.canvas = Canvas(self.root, width=400, height=400, bg="#ecf0f1", highlightthickness=2, highlightbackground="#bdc3c7")
        self.canvas.pack(pady=10)
        
        msg = "DROP IMAGE HERE" if self.models_loaded else "FIX MISSING FILES"
        self.instruction_text = self.canvas.create_text(200, 200, text=msg, 
                                                       justify=CENTER, font=("Arial", 14, "bold"), fill="#7f8c8d")

        self.res_var = StringVar(value="Waiting for images...")
        self.lbl_res = Label(self.root, textvariable=self.res_var, fg="#ecf0f1", bg="#2c3e50", font=("Arial", 18, "bold"))
        self.lbl_res.pack(pady=20)

        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind('<<Drop>>', self.on_image_drop)

    def extract_features(self, img):
        roi = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        c_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(c_hist, c_hist)
        c_feat = c_hist.flatten()

        gray = hsv[:,:,2] 
        radius, n_points = 3, 24
        h, w = gray.shape
        h_step, w_step = h // GRID_SIZE, w // GRID_SIZE
        l_feat = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = gray[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                lbp = local_binary_pattern(cell, n_points, radius, method="uniform")
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
                l_feat.extend(hist.astype("float") / (hist.sum() + 1e-7))
        
        return np.hstack((c_feat, l_feat)).reshape(1, -1)

    def on_image_drop(self, event):
        if not self.models_loaded: return
        path = event.data.strip('{}')
        if not os.path.isfile(path): return

        try:
            pil_img = Image.open(path)
            pil_img.thumbnail((400, 400))
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("all")
            self.canvas.create_image(200, 200, image=self.tk_img)

            self.res_var.set("Processing...")
            self.root.update_idletasks()

            cv_img = cv2.imread(path)
            feats = self.extract_features(cv_img)
            scaled = self.scaler.transform(feats)
            pca_feats = self.pca.transform(scaled)
            
            pred = int(self.clf.predict(pca_feats)[0])
            probs = self.clf.predict_proba(pca_feats)[0] if hasattr(self.clf, "predict_proba") else None
            conf = f" ({max(probs)*100:.1f}%)" if probs is not None else ""
            
            result_text = OBJECT_NAMES.get(pred, f"Class {pred}")
            self.res_var.set(f"Result: {result_text}{conf}")
            
        except Exception as e:
            print(f"Error: {e}")
            self.res_var.set("Error processing image")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = InstantDetector(root)
    root.mainloop()