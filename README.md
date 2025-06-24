# PostureGuardian: AI-Based Rehab Posture Checker

**PostureGuardian** is a Python-based mini-project that uses AI to classify arm raise exercise postures as **Correct** or **Incorrect** in real-time using webcam input. It uses **MediaPipe** for pose estimation, a **Random Forest** classifier for classification, and **OpenCV** for visualization, trained on a subset of the **WLU Rehabilitation Posture Dataset**.

---

## Project Overview

- **Objective**: Build a real-time AI system to provide feedback on arm raise exercise posture.
- **Focus**: Human pose estimation, machine learning, and live classification for rehabilitation support.
- **Dataset**: 50 videos (25 Correct, 25 Incorrect), split 80% train, 10% validation, 10% test.

### Features

- Extracts pose landmarks using MediaPipe Pose
- Computes normalized landmarks, elbow angles, and shoulder-to-wrist distances
- Trains Random Forest classifier for classification
- Real-time feedback via webcam with arm-focused skeleton overlay and posture label

---

## Requirements

- Python 3.8+
- Libraries:
  - `opencv-python`
  - `mediapipe`
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `tqdm`
  - `matplotlib`
  - `seaborn`
  - `joblib`

### Install dependencies

```bash
pip install opencv-python mediapipe scikit-learn numpy pandas tqdm matplotlib seaborn joblib
```

---

## Project Structure

```plaintext
dataset_small/
├── train/
│   ├── Arm Raise Correct/
│   └── Arm Raise Incorrect/
├── val/
│   ├── Arm Raise Correct/
│   └── Arm Raise Incorrect/
└── test/
    ├── Arm Raise Correct/
    └── Arm Raise Incorrect/

postureguardian.ipynb           # Main notebook with all code
postureguardian_model.joblib    # Trained Random Forest model
preprocessed_data.npz           # Preprocessed dataset features and labels
README.md                       # Project documentation
```

---

## Setup and Usage

### Clone the Repository

```bash
git clone https://github.com/povsalman/PostureGuardian-AI-Based-Rehab-Posture-Checker.git
cd postureguardian
```

### Prepare Dataset

Place the `dataset_small` folder in the project root as shown in the structure above.

### Run the Notebook

1. Open `postureguardian.ipynb` in Jupyter Notebook.
2. Execute cells in order:

   - **Cell 1**: Import libraries and initialize MediaPipe
   - **Cell 2**: Define helper functions for feature extraction and visualization
   - **Cell 3**: Preprocess dataset with optimal frame sampling (frame skip = 5)
   - **Cell 4**: Train Random Forest classifier and save best model
   - **Cell 5**: Evaluate model (accuracy, precision, recall, F1, confusion matrix)
   - **Cell 6**: Run real-time webcam demo (press 'q' to exit)

### Outputs

- Preprocessed features: `preprocessed_data.npz`
- Trained model: `postureguardian_model.joblib`
- Webcam demo visuals: screenshots/video (manual capture)

---

## Implementation Details

### Preprocessing

- Extract MediaPipe landmarks (x, y, z, visibility) from every next frame
- Normalize coordinates using torso midpoint and shoulder width
- Add elbow angles and shoulder-to-wrist distances

### Model

- Random Forest classifier with grid search:
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [5, 10, None]
- Feature vector: 136-dimensional  
  (33 keypoints × 4 + 2 angles + 2 distances)

### Evaluation

- Metrics: Accuracy, Precision, Recall, F1-score
- Plots: Confusion matrix, metric bar chart

### Real-Time Inference

- Webcam frame processing with 10-frame prediction averaging
- Overlays "Correct" (green) or "Incorrect" (red) label with skeleton visualization

---

## Performance

- **Validation Accuracy**: ~80%
- **Test Metrics **:

- **Frame Skip**: Chosen as 5 for ~36 frames per 6-sec video

---

## Challenges

- Normalizing for varying body sizes and distances
- Stabilizing real-time predictions using moving average
- Balancing frame sampling for speed vs accuracy

---

## Deliverables

- Jupyter Notebook: `postureguardian.ipynb`
- Trained model: `postureguardian_model.joblib`
- Demo visuals (manual capture)
- Documentation in notebook and `README.md`

---

## Future Improvements

- Add more joint angle and torso alignment features
- Enable audio alerts for incorrect posture
- Save prediction logs to CSV for long-term analysis

---
