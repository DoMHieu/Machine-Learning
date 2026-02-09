import sys
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QCursor

MODEL_IMG_SIZE = 128
GRID_SIZE = 4
CONFIDENCE_THRESHOLD = 0.95
STEP_SIZE = 32
SCALES = [1.0, 0.75, 0.5]
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

def extract_features(patch):
    patch = cv2.resize(patch, (MODEL_IMG_SIZE, MODEL_IMG_SIZE))
    c_feat = get_color_features(patch)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:,:,2]
    l_feat = get_spatial_lbp_features(gray)
    return np.hstack((c_feat, l_feat))

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0: return []
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
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
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

#UI CLASSES
class ImageLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("\nImport Image\n")
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                background-color: #f0f0f0;
                color: #555;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel:hover {
                background-color: #e0e0e0;
                border-color: #888;
            }
        """)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.parent_app = parent

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent_app.open_file_dialog()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.parent_app.load_image(file_path)
            else:
                QMessageBox.warning(self, "Error", "Please drop image files only!")

class FireDetectApp(QMainWindow): #UI COOK
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dumb testing app")
        self.setGeometry(100, 100, 1000, 750)
        self.model = None
        self.scaler = None
        self.pca = None
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.image_label = ImageLabel(self)
        self.image_label.setMinimumSize(800, 500)
        main_layout.addWidget(self.image_label)
        self.btn_detect = QPushButton("Object Detection")
        self.btn_detect.setFixedHeight(60)
        self.btn_detect.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.btn_detect.clicked.connect(self.run_detection)
        self.btn_detect.setEnabled(False)
        main_layout.addWidget(self.btn_detect)

        self.status_label = QLabel("Initializing...")
        self.statusBar().addWidget(self.status_label)

        self.current_image = None
        self.load_models()

    def load_models(self):
        try:
            print("Loading model...")
            model_path = 'new_model_svm_cuml.pkl'
            self.model = joblib.load(model_path)
            self.scaler = joblib.load('model_scaler.pkl')
            try:
                self.pca = joblib.load('model_pca.pkl')
                self.status_label.setText(f"✅ Ready (Model: {model_path} + Scaler + PCA)")
            except:
                self.pca = None
                self.status_label.setText(f"✅ Ready (Model: {model_path} + Scaler)")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"Model file not found!\n\n{e}")
            self.status_label.setText("❌ Model error")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.status_label.setText("❌ Model error")

    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Images (*.jpg *.png *.jpeg)")
        if fname: self.load_image(fname)

    def load_image(self, path):
        image = cv2.imread(path)
        if image is None: return
        self.current_image = image
        self.display_image(image)
        self.btn_detect.setEnabled(True)
        self.status_label.setText(f"Loaded: {path}")

    def display_image(self, img_bgr):
        disp_img = img_bgr.copy()
        if disp_img.shape[1] > 1000:
            scale = 1000 / disp_img.shape[1]
            disp_img = cv2.resize(disp_img, None, fx=scale, fy=scale)
        img_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qt_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def run_detection(self):
        if self.current_image is None or self.model is None: return
        self.status_label.setText("⏳ Scanning image...")
        self.btn_detect.setEnabled(False)
        QApplication.processEvents()
        raw_img = self.current_image.copy()
        output_img = raw_img.copy()
        detected_fire = []
        detected_smoke = []
        win_size = (MODEL_IMG_SIZE, MODEL_IMG_SIZE)
        for scale in SCALES:
            width = int(raw_img.shape[1] * scale)
            height = int(raw_img.shape[0] * scale)
            resized = cv2.resize(raw_img, (width, height))
            if resized.shape[0] < win_size[1] or resized.shape[1] < win_size[0]: continue
            for (x, y, patch) in sliding_window(resized, STEP_SIZE, win_size):
                features = extract_features(patch).reshape(1, -1)
                features = self.scaler.transform(features)
                if self.pca: features = self.pca.transform(features)
                probas = self.model.predict_proba(features)[0]
                label = np.argmax(probas)
                score = np.max(probas)
                if score > CONFIDENCE_THRESHOLD:
                    real_x = int(x / scale)
                    real_y = int(y / scale)
                    real_w = int(win_size[0] / scale)
                    real_h = int(win_size[1] / scale)
                    if label == 2:
                        detected_fire.append([real_x, real_y, real_x + real_w, real_y + real_h])
                    elif label == 1:
                        detected_smoke.append([real_x, real_y, real_x + real_w, real_y + real_h])
        count_fire = 0
        count_smoke = 0
        if len(detected_fire) > 0:
            boxes = non_max_suppression(np.array(detected_fire), 0.3)
            count_fire = len(boxes)
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(output_img, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if len(detected_smoke) > 0:
            boxes = non_max_suppression(np.array(detected_smoke), 0.3)
            count_smoke = len(boxes)
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(output_img, "SMOKE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        self.display_image(output_img)
        self.btn_detect.setEnabled(True)
        msg = f"Done. Found {count_fire} Fire, {count_smoke} Smoke."
        self.status_label.setText(msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FireDetectApp()
    window.show()
    sys.exit(app.exec_())