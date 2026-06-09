"""
train_models.py — Local training script for the Autonomous Navigation Drone project.

Runs the full pipeline locally (no Colab / no yt-dlp required) using an existing
`video.mp4` in this folder, and generates all trained model files:

    model_cnn_baseline.h5
    model_cnn_rnn.h5
    model_cnn_lstm.h5
    model_cnn_gru.h5
    model_cnn_transformer.h5
    model_drone_info.h5

plus the cached arrays (X.npy, y.npy, features) the Streamlit app uses for the
Models Comparison section.

Usage:
    python train_models.py
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEQUENCE_LENGTH = 5
NAV_IMG_SIZE = (224, 224)
NUM_CLASSES = 4
MAX_SAMPLES = 800

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


# =====================================================
# Custom layers (with get_config so the app can reload them)
# =====================================================

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos_embed = self.pos_embedding(positions)
        return x + pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "embed_dim": self.embed_dim})
        return config


# =====================================================
# Step 1: Extract frames + build sequences (brightness pseudo-labels)
# =====================================================

def build_dataset():
    if os.path.exists("X.npy") and os.path.exists("y.npy"):
        print("Found cached X.npy / y.npy — skipping frame extraction.")
        return np.load("X.npy"), np.load("y.npy")

    if not os.path.exists("video.mp4"):
        raise FileNotFoundError("video.mp4 not found in project folder. Add it first.")

    print("Extracting frames from video.mp4 ...")
    video = cv2.VideoCapture("video.mp4")
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frames.append(frame)
    video.release()
    print(f"  Extracted {len(frames)} frames")

    X, y = [], []
    frame_skip = 2
    for i in range(0, len(frames) - SEQUENCE_LENGTH, frame_skip):
        seq = []
        for j in range(SEQUENCE_LENGTH):
            img = cv2.resize(frames[i + j], NAV_IMG_SIZE) / 255.0
            seq.append(img)
        if len(seq) != SEQUENCE_LENGTH:
            continue
        X.append(seq)

        last = (seq[-1] * 255).astype(np.uint8)
        gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        left, right, overall = gray[:, :w // 2].mean(), gray[:, w // 2:].mean(), gray.mean()
        if overall < 40:
            label = 3          # Stop
        elif abs(left - right) < 5:
            label = 2          # Straight
        elif left < right:
            label = 1          # Right
        else:
            label = 0          # Left
        y.append(label)

    X = np.array(X, dtype="float32")[:MAX_SAMPLES]
    y = np.array(y)[:MAX_SAMPLES]
    np.save("X.npy", X)
    np.save("y.npy", y)
    print(f"  X={X.shape}  y={y.shape}  classes={np.bincount(y)}")
    return X, y


# =====================================================
# Step 2: MobileNetV2 feature extraction (batched)
# =====================================================

def extract_features(X_data, cnn):
    n, s = X_data.shape[0], X_data.shape[1]
    flat = X_data.reshape(-1, *X_data.shape[2:])
    feats = cnn.predict(flat, batch_size=32, verbose=0)
    return feats.reshape(n, s, -1)


# =====================================================
# Navigation model builders
# =====================================================

def build_cnn_baseline():
    inp = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out)


def build_recurrent(kind, seq_len, feat_dim):
    inp = layers.Input(shape=(seq_len, feat_dim))
    if kind == 'rnn':
        x = layers.SimpleRNN(128)(inp)
    elif kind == 'lstm':
        x = layers.LSTM(128)(inp)
    else:
        x = layers.GRU(128)(inp)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out)


def build_transformer(seq_len, feat_dim, num_heads=4, ff_dim=256):
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.Dense(128)(inp)
    x = PositionalEncoding(seq_len, 128)(x)
    x = TransformerBlock(128, num_heads, ff_dim)(x)
    x = TransformerBlock(128, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out)


# =====================================================
# Drone info model (synthetic dataset fallback)
# =====================================================

def build_drone_model(n_type=4, n_speed=3, n_price=3):
    inp = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    t = layers.Dense(n_type, activation='softmax', name='type_output')(layers.Dense(64, activation='relu')(x))
    s = layers.Dense(n_speed, activation='softmax', name='speed_output')(layers.Dense(64, activation='relu')(x))
    p = layers.Dense(n_price, activation='softmax', name='price_output')(layers.Dense(64, activation='relu')(x))
    return Model(inp, [t, s, p])


def make_synthetic_drones(n=200):
    np.random.seed(42)
    type_cats = ['Combat', 'Multi-purpose', 'Surveillance', 'Transport']
    speed_cats = ['High', 'Low', 'Medium']
    price_cats = ['Consumer ($1K-100K)', 'Military ($1M+)', 'Professional ($100K-1M)']
    imgs, ts, ss, ps = [], [], [], []
    for i in range(n):
        t_idx, s_idx, p_idx = i % 4, i % 3, i % 3
        img = np.random.rand(128, 128, 3) * 0.2
        if t_idx == 0:
            img[:, :, 0] += 0.3; img[30:90, 20:108, :] += 0.2
        elif t_idx == 1:
            img += 0.25; img[40:88, 40:88, :] += 0.3
        elif t_idx == 2:
            img[:, :, 2] += 0.3; img[50:78, 10:118, :] += 0.2
        else:
            img += 0.15; img[20:108, 35:93, :] += 0.3
        if s_idx == 0:
            img += np.linspace(0, 0.3, 128).reshape(1, 128, 1)
        elif s_idx == 2:
            img += np.random.rand(128, 128, 3) * 0.1
        if p_idx == 1:
            img += np.random.rand(128, 128, 3) * 0.15
        imgs.append(np.clip(img, 0, 1).astype(np.float32))
        ts.append(type_cats[t_idx]); ss.append(speed_cats[s_idx]); ps.append(price_cats[p_idx])
    X = np.array(imgs)
    y_type = LabelEncoder().fit_transform(ts)
    y_speed = LabelEncoder().fit_transform(ss)
    y_price = LabelEncoder().fit_transform(ps)
    return X, y_type, y_speed, y_price


# =====================================================
# Main pipeline
# =====================================================

def main():
    X, y = build_dataset()
    y_cat = to_categorical(y, NUM_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    # ---- CNN baseline (last frame) ----
    print("\n[1/6] Training CNN Baseline ...")
    m = build_cnn_baseline()
    m.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    m.fit(X_train[:, -1], y_train, validation_data=(X_test[:, -1], y_test),
          epochs=15, batch_size=16, verbose=2)
    m.save('model_cnn_baseline.h5')

    # ---- MobileNetV2 features ----
    print("\nExtracting MobileNetV2 features ...")
    cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    cnn.trainable = False
    X_train_feat = extract_features(X_train, cnn)
    X_test_feat = extract_features(X_test, cnn)
    np.save("X_train_feat.npy", X_train_feat)
    np.save("X_test_feat.npy", X_test_feat)
    seq_len, feat_dim = X_train_feat.shape[1], X_train_feat.shape[2]

    # ---- Recurrent models ----
    for tag, name, epochs in [('rnn', '[2/6] CNN+RNN', 20),
                              ('lstm', '[3/6] CNN+LSTM', 20),
                              ('gru', '[4/6] CNN+GRU', 20)]:
        print(f"\n{name} ...")
        rm = build_recurrent(tag, seq_len, feat_dim)
        rm.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        rm.fit(X_train_feat, y_train, validation_data=(X_test_feat, y_test),
               epochs=epochs, batch_size=16, verbose=2)
        rm.save(f'model_cnn_{tag}.h5')

    # ---- Transformer ----
    print("\n[5/6] CNN+Transformer ...")
    tm = build_transformer(seq_len, feat_dim)
    tm.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    tm.fit(X_train_feat, y_train, validation_data=(X_test_feat, y_test),
           epochs=25, batch_size=16, verbose=2)
    tm.save('model_cnn_transformer.h5')

    # ---- Drone info (synthetic) ----
    print("\n[6/6] Drone Info model (synthetic dataset) ...")
    Xd, yt, ys, yp = make_synthetic_drones(200)
    Xd_tr, Xd_te, yt_tr, yt_te, ys_tr, ys_te, yp_tr, yp_te = train_test_split(
        Xd, yt, ys, yp, test_size=0.2, random_state=42
    )
    dm = build_drone_model()
    dm.compile('adam',
               loss={'type_output': 'sparse_categorical_crossentropy',
                     'speed_output': 'sparse_categorical_crossentropy',
                     'price_output': 'sparse_categorical_crossentropy'},
               metrics={'type_output': 'accuracy', 'speed_output': 'accuracy', 'price_output': 'accuracy'})
    dm.fit(Xd_tr, {'type_output': yt_tr, 'speed_output': ys_tr, 'price_output': yp_tr},
           validation_data=(Xd_te, {'type_output': yt_te, 'speed_output': ys_te, 'price_output': yp_te}),
           epochs=30, batch_size=16, verbose=2)
    dm.save('model_drone_info.h5')

    print("\nDone. All 6 model files saved in:", HERE)


if __name__ == "__main__":
    main()
