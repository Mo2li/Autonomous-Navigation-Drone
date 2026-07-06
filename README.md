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

| Model | Architecture | Val. Accuracy\* |
|-------|-------------|----------|
| CNN + LSTM | MobileNetV2 + LSTM(128) | ~83% 🥇 |
| CNN + GRU | MobileNetV2 + GRU(128) | ~83% 🥇 |
| CNN + RNN | MobileNetV2 + SimpleRNN(128) | ~82% |
| CNN + Transformer | MobileNetV2 + Multi-Head Attention | ~80% |
| CNN Baseline | Conv2D only (single frame) | ~57% |

> **Key finding (relative):** The recurrent sequence models (LSTM/GRU) consistently outperform the single-frame CNN baseline, which supports the core hypothesis — **temporal context matters** for navigation prediction.

> \* ⚠️ **Read these as relative, not absolute.** Because the frames come from a *single continuous video* and the train/test split is random (not temporal), adjacent near-duplicate frames can appear in both sets — i.e. **temporal data leakage inflates the raw numbers**. The comparison *between* models is meaningful (all share the same split); the absolute accuracy is **not** a production-grade metric. See [Limitations](#-limitations).

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

- **Source:** Frames extracted from a **single** continuous navigation video
- **Labeling:** Pseudo-labels generated using a frame-brightness heuristic (not human-annotated)
- **Input:** Sequences of 5 consecutive frames (224×224×3)
- **Split:** 80% train / 20% test — **random** split (not time-based)

> ⚠️ This is a **proof-of-concept / learning project**, not a validated production model. Two things limit the metrics: (1) pseudo-labels are heuristic, and (2) the random split over one continuous video causes temporal leakage. Treat the results as a comparative study of architectures, not a real-world accuracy claim.

---

## 🔍 Key Findings

1. **The single-frame CNN baseline lags well behind** the sequence models — as expected, one frame carries no motion information.
2. **LSTM and GRU lead** the comparison, capturing short-term motion patterns better than the other variants.
3. **The Transformer** is competitive overall but struggles on the `Straight` class, likely due to class imbalance and the small dataset.
4. **Macro F1-score** is more informative than raw accuracy here, given the class imbalance.

> Again: these are **relative** conclusions across models sharing the same split — not absolute performance claims (see Limitations).

---

## ⚠️ Limitations

- **Temporal leakage:** frames come from one continuous video and the split is random, so near-identical frames can land in both train and test — this inflates the reported accuracy.
- **Heuristic labels:** brightness-based pseudo-labels approximate direction; they are not ground-truth control commands.
- **Single-source data:** one video limits generalization; the model would not transfer to unseen environments as-is.
- **What this project *does* show:** a clean multi-architecture comparison pipeline (CNN → RNN/LSTM/GRU → Transformer), feature extraction with a frozen MobileNetV2, and end-to-end deployment via Streamlit.

**To make it production-grade:** use a time-based split across *multiple* videos, real control-command labels, and report precision/recall/F1 per class.

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
