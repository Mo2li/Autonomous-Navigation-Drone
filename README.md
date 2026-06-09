# 🚁 Autonomous Navigation Drone — Multi-Model Deep Learning System

A deep learning project that compares **5 sequential model architectures** for autonomous drone navigation prediction from video sequences, plus a multi-output CNN for drone information analysis.

---

## 📌 Project Overview

The system takes a short video sequence as input and predicts the drone's next navigation action:

| Class | Direction |
|-------|-----------|
| 0 | ⬅️ Left |
| 1 | ➡️ Right |
| 2 | ⬆️ Straight |
| 3 | 🛑 Stop |

---

## 🧠 Models Compared

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| CNN + LSTM | MobileNetV2 + LSTM(128) | **83.13%** 🥇 |
| CNN + GRU | MobileNetV2 + GRU(128) | **83.13%** 🥇 |
| CNN + RNN | MobileNetV2 + SimpleRNN(128) | 81.88% |
| CNN + Transformer | MobileNetV2 + Multi-Head Attention | 80.00% |
| CNN Baseline | Conv2D only (single frame) | 56.88% |

> **Key finding:** Sequence models significantly outperform the CNN baseline, confirming that temporal information is critical for navigation prediction.

---

## 🏗️ Architecture

```
Video → Frame Extraction → Sequences of 5 frames
                                    ↓
                        MobileNetV2 (frozen, ImageNet)
                        Feature Extraction: 1280-dim per frame
                                    ↓
                    ┌───────────────────────────────┐
                    │  RNN / LSTM / GRU / Transformer│
                    └───────────────────────────────┘
                                    ↓
                        Dense(64) → Dropout(0.3)
                                    ↓
                            Softmax(4 classes)
                        Left / Right / Straight / Stop
```

---

## 🚀 Features

### 1. Navigation System (Video Input)
- Upload any video or use the scraped dataset video
- Choose from 5 model architectures
- Visualize frame-by-frame predictions
- Direction distribution pie chart

### 2. Drone Info Analyzer (Image Input)
- Upload a drone image
- Multi-output CNN predicts:
  - **Type:** Combat / Multi-purpose / Surveillance / Transport
  - **Speed:** High / Medium / Low
  - **Price Range:** Consumer / Professional / Military

### 3. Models Comparison Dashboard
- Full benchmark on test set
- Accuracy, Precision, Recall, F1-Score table
- Bar chart & Radar chart comparison
- Per-model confusion matrices

---

## 📁 Project Structure

```
Autonomous-Navigation-Drone/
├── Autonomous_Navigation_Drone_Final.ipynb          # Main training notebook (Colab)
├── app.py                                           # Streamlit web application
├── train_models.py                                  # Local training script (no Colab needed)
├── model_cnn_baseline.h5                            # ┐
├── model_cnn_rnn.h5                                 # │
├── model_cnn_lstm.h5                                # │ Pre-trained models
├── model_cnn_gru.h5                                 # │ (app runs out-of-the-box)
├── model_cnn_transformer.h5                         # │
├── model_drone_info.h5                             # ┘
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The trained `.h5` models are included, so the app runs immediately after `pip install`.
> Large data/video files (`.npy`, `.mp4`) are excluded — regenerate them with `train_models.py`.

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Mo2li/Autonomous-Navigation-Drone.git
cd Autonomous-Navigation-Drone
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
The pre-trained models are included, so you can run the app right away:
```bash
streamlit run app.py
```

### (Optional) Re-train the models
Two options:

**A) Locally — no Colab needed** (uses an existing `video.mp4` in the folder):
```bash
python train_models.py
```
This regenerates all 6 `.h5` model files plus the cached `.npy` arrays.

**B) On Google Colab** — run `Autonomous_Navigation_Drone_Final.ipynb`, which additionally
scrapes the video dataset from YouTube via `yt-dlp` and downloads drone images.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Pretrained CNN | MobileNetV2 (ImageNet) |
| Computer Vision | OpenCV |
| Web App | Streamlit |
| Visualization | Plotly, Matplotlib |
| Data | NumPy, Pandas |
| Evaluation | scikit-learn |

---

## 📊 Dataset

- **Source:** Video frames extracted from a drone navigation video
- **Labeling:** Pseudo-labels generated using frame brightness heuristics
- **Input:** Sequences of 5 consecutive frames (224×224×3)
- **Split:** 80% train / 20% test

> ⚠️ Labels are pseudo-labels (brightness-based heuristic), not human-annotated. This is suitable for a course prototype.

---

## 🔍 Key Findings

1. **CNN baseline (56.88%)** is limited because it uses only the last frame with no temporal context.
2. **LSTM and GRU (83.13%)** are the best performers — they capture short-term motion patterns effectively.
3. **Transformer (80.00%)** shows strong overall accuracy but fails on the `Straight` class due to class imbalance and small dataset size.
4. **Macro F1-score** is more informative than accuracy alone for this imbalanced dataset.

---

## 🔮 Future Improvements

- Use real drone control commands instead of pseudo-labels
- Add optical flow or motion vectors as extra features
- Apply class weighting or oversampling for balanced training
- Increase dataset size and diversity
- Explore ViT or TimeSformer for video understanding
- Deploy the Streamlit app to Streamlit Cloud

---

## 👤 Author

**Muhammed Ali** — Computer Science student at Helwan University, specializing in AI & Data Science.

[![GitHub](https://img.shields.io/badge/GitHub-Mo2li-black?logo=github)](https://github.com/Mo2li)
