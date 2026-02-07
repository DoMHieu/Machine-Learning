# Machine-Learning
A simple machine learning model that can detect fire/smoke from picture

Build 2 (Sort of) model, one train the data raw on a 5k image, that take hours and hours to train
The other optimized one take few minutes to train with PCA optimization and store to Cache

The dataset file use for this that have been cleaned:
https://drive.google.com/file/d/1TCiCWTQbI_8jFmKb_L4OrnR3Vr4K4n66/view?usp=sharing

The original dataset:
https://www.kaggle.com/datasets/sarthaktandulje/disaster-damage-5class

# New model using labeled dataset:
https://github.com/gaia-solutions-on-demand/DFireDataset
Use this dataset and put it in the same folder with train.py and preprocessing.py
train.py uses Cuml which required setup wsl and conda. If you want to train on sklearn, then just change the code. (Just do it on SKlearn, it so much better for your mental health)

# New model working
1. Preprocessing:
    - Read file from dataset:
        Positive sample: Cut Region of Interest (ROI) from dataset image, then resize to 128x128. then class object to 1 or 2 (fire or smoke)
        Negative sample: random crop outside of ROI, resize to 128x128, class to 0.
    - Feature Extraction:
        Color: Convert ROI to HSV. (8x8x8)
        Texture: Using LBP, divide into 4x4 grid.
        Combline color and texture.
    - StandardScaler
    - Using PCA to reduce vector (I set reduce to 512(He lied it 912 for some reason))
    - Save to .npy 
    - Side Note: It take an eternity for it to finished, so be prepare.
2. Train:
    - (To train in SKlearn, you can do so by changing all the import that have Cuml to Sklearn) (Side Note: It take at least 10 minutes or more to train in SKlearn, but setting up using Cuml is also a pain)
    - Load .npy saved by preprocessing.
    - Training with SVM:
        Using RBF kernel
        C = 1000 (Prioritize in reducing training error)
        gamma = scale
        probabiliy = True
        cache_size = 2000 (More = better = require better hardware)

3. Running (Test App)
    - The "main_run.py" file is just a pure running model script that just checking if the image have fire, smoke or not.
    - The "detect_run.py" is a script that use Sliding Window technique to check the image for "where" the fire or smoke is.
    - Both script is written in Sk-learn not cuml, so change the training script to sk-learn and train again with Sk-learn or the mmodel will not work.
    - Need someone to optimized or rewritten the entirety of the trainning with label for it to work better at detecting object.
    - (Optional) Rewritten both script for it to work with Cuml preferbaly.