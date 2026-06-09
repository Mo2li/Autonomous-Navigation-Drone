import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import shutil

# Always resolve model/data files relative to this file's folder,
# regardless of the working directory the app is launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# Custom Layers (must be defined before loading models)
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
        config.update({
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
        })
        return config


# =====================================================
# Constants
# =====================================================

CLASS_NAMES = ['Left', 'Right', 'Straight', 'Stop']
SEQUENCE_LENGTH = 5
NAV_IMG_SIZE = (224, 224)
DRONE_IMG_SIZE = (128, 128)

TYPE_CLASSES = np.array(['Combat', 'Multi-purpose', 'Surveillance', 'Transport'])
SPEED_CLASSES = np.array(['High', 'Low', 'Medium'])
PRICE_CLASSES = np.array(['Consumer ($1K-100K)', 'Military ($1M+)', 'Professional ($100K-1M)'])

CUSTOM_OBJECTS = {
    'TransformerBlock': TransformerBlock,
    'PositionalEncoding': PositionalEncoding,
}

MODEL_FILES = {
    'CNN (Baseline)': 'model_cnn_baseline.h5',
    'CNN + RNN': 'model_cnn_rnn.h5',
    'CNN + LSTM': 'model_cnn_lstm.h5',
    'CNN + GRU': 'model_cnn_gru.h5',
    'CNN + Transformer': 'model_cnn_transformer.h5',
}

DIRECTION_EMOJIS = {
    'Left': '⬅️',
    'Right': '➡️',
    'Straight': '⬆️',
    'Stop': '🛑',
}

# =====================================================
# Model loading (cached)
# =====================================================

@st.cache_resource
def load_feature_extractor():
    """Load MobileNetV2 for CNN feature extraction."""
    base_cnn = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3), pooling='avg')
    base_cnn.trainable = False
    return base_cnn


@st.cache_resource
def load_nav_model(model_name):
    """Load a navigation model by name."""
    path = MODEL_FILES[model_name]
    if not os.path.exists(path):
        return None
    model = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
    return model


@st.cache_resource
def load_drone_model():
    """Load the drone info multi-output model."""
    path = 'model_drone_info.h5'
    if not os.path.exists(path):
        return None
    model = tf.keras.models.load_model(path)
    return model


# =====================================================
# Processing functions (exact same as notebook)
# =====================================================

def extract_frames_from_video(video_path):
    """Extract all frames from a video file."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    video.release()
    return frames, fps, total


def build_sequences(frames, sequence_length=5, frame_skip=2):
    """Build sequences from frames exactly like the notebook."""
    X = []
    for i in range(0, len(frames) - sequence_length, frame_skip):
        seq = []
        for j in range(sequence_length):
            img = cv2.resize(frames[i + j], NAV_IMG_SIZE)
            img = img / 255.0
            seq.append(img)
        if len(seq) == sequence_length:
            X.append(seq)
    return np.array(X, dtype="float32")


def extract_features(X_data, cnn_model):
    """Extract features using MobileNetV2 — batched for speed."""
    n_samples, seq_len = X_data.shape[0], X_data.shape[1]
    flat = X_data.reshape(-1, *X_data.shape[2:])  # (n*seq_len, H, W, C)
    features_flat = cnn_model.predict(flat, batch_size=32, verbose=0)
    return features_flat.reshape(n_samples, seq_len, -1)


def predict_navigation_on_sequences(X_seq, model, model_name, cnn_extractor):
    """Run navigation prediction on sequences."""
    if model_name == 'CNN (Baseline)':
        # CNN baseline uses only last frame
        X_input = X_seq[:, -1]  # (samples, 224, 224, 3)
        predictions = model.predict(X_input, verbose=0)
    else:
        # Other models use extracted features
        features = extract_features(X_seq, cnn_extractor)
        predictions = model.predict(features, verbose=0)
    return predictions


def predict_drone_info(image, model):
    """Predict drone info from a PIL image."""
    img = image.convert('RGB').resize(DRONE_IMG_SIZE)
    img_arr = img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_arr, axis=0)
    type_pred, speed_pred, price_pred = model.predict(img_batch, verbose=0)
    return {
        'Type': TYPE_CLASSES[np.argmax(type_pred)],
        'Type Confidence': float(np.max(type_pred) * 100),
        'Speed': SPEED_CLASSES[np.argmax(speed_pred)],
        'Speed Confidence': float(np.max(speed_pred) * 100),
        'Price': PRICE_CLASSES[np.argmax(price_pred)],
        'Price Confidence': float(np.max(price_pred) * 100),
    }


# =====================================================
# Streamlit App
# =====================================================

st.set_page_config(
    page_title="Autonomous Navigation Drone",
    page_icon="🚁",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🚁 Autonomous Navigation Drone</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Multimodel System for Sequence Learning Navigation + Drone Info Analyzer</h4>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# Sidebar
# =====================================================

section = st.sidebar.radio(
    "📌 Choose Section",
    ["🎬 Navigation System (Video)", "📷 Drone Info Analyzer (Image)", "📊 Models Comparison"],
    index=0,
)

# =====================================================
# SECTION 1: Navigation System
# =====================================================

if section == "🎬 Navigation System (Video)":
    st.header("🎬 Navigation System — Video → Direction Prediction")
    st.markdown("Predict drone navigation directions (**Left / Right / Straight / Stop**) from video sequences.")

    # Choose model
    model_choice = st.selectbox(
        "🧠 Choose Navigation Model",
        list(MODEL_FILES.keys()),
        index=4,  # default to Transformer
    )

    # Show model architecture info
    with st.expander("ℹ️ Model Architecture Details"):
        if model_choice == 'CNN (Baseline)':
            st.markdown("""
            **CNN Only (Baseline)**
            - Uses only the **last frame** of each 5-frame sequence
            - Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
            - GlobalAveragePooling2D → Dense(128) → Dropout(0.3) → Softmax(4)
            - *No temporal/sequence modeling*
            """)
        elif model_choice == 'CNN + RNN':
            st.markdown("""
            **CNN + SimpleRNN**
            - Feature extraction: MobileNetV2 (frozen, ImageNet weights) → 1280-dim features per frame
            - Sequence modeling: SimpleRNN(128)
            - Dense(64) → Dropout(0.3) → Softmax(4)
            """)
        elif model_choice == 'CNN + LSTM':
            st.markdown("""
            **CNN + LSTM**
            - Feature extraction: MobileNetV2 (frozen, ImageNet weights) → 1280-dim features per frame
            - Sequence modeling: LSTM(128)
            - Dense(64) → Dropout(0.3) → Softmax(4)
            """)
        elif model_choice == 'CNN + GRU':
            st.markdown("""
            **CNN + GRU**
            - Feature extraction: MobileNetV2 (frozen, ImageNet weights) → 1280-dim features per frame
            - Sequence modeling: GRU(128)
            - Dense(64) → Dropout(0.3) → Softmax(4)
            """)
        elif model_choice == 'CNN + Transformer':
            st.markdown("""
            **CNN + Transformer (Most Advanced)**
            - Feature extraction: MobileNetV2 (frozen, ImageNet weights) → 1280-dim features per frame
            - Dense projection to 128-dim
            - Positional Encoding (learned embeddings)
            - 2× Transformer Blocks (MultiHeadAttention 4 heads + FFN 256)
            - GlobalAveragePooling1D → Dense(64) → Dropout(0.3) → Softmax(4)
            """)

    st.markdown("---")

    # Video source
    video_source = st.radio(
        "📂 Video Source",
        ["🎥 Use Scraped Video (video.mp4)", "📤 Upload Your Own Video"],
        horizontal=True,
    )

    video_path = None

    if video_source == "🎥 Use Scraped Video (video.mp4)":
        if os.path.exists("video.mp4"):
            video_path = "video.mp4"
            st.video("video.mp4")
            st.success("✅ Scraped video loaded successfully!")
        else:
            st.error("⚠️ video.mp4 not found! Please run the notebook first to download it, or upload your own video.")
    else:
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            # Save uploaded video to temp file
            tmp_dir = tempfile.mkdtemp()
            video_path = os.path.join(tmp_dir, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            st.video(uploaded_video)
            st.success("✅ Video uploaded successfully!")

    if video_path and st.button("🚀 Run Navigation Prediction", type="primary"):
        # Load model
        nav_model = load_nav_model(model_choice)
        if nav_model is None:
            st.error(f"⚠️ Model file '{MODEL_FILES[model_choice]}' not found! Train and save models first.")
        else:
            cnn_extractor = load_feature_extractor()

            with st.spinner("📹 Extracting frames from video..."):
                frames, fps, total_frames = extract_frames_from_video(video_path)
                st.info(f"📊 Video: FPS={fps:.1f}, Total Frames={total_frames}, Extracted={len(frames)}")

            if len(frames) < SEQUENCE_LENGTH:
                st.error("⚠️ Not enough frames in video!")
            else:
                with st.spinner("🔧 Building sequences..."):
                    X_seq = build_sequences(frames, SEQUENCE_LENGTH, frame_skip=2)
                    st.info(f"📦 Built {X_seq.shape[0]} sequences of {SEQUENCE_LENGTH} frames each")

                with st.spinner(f"🧠 Running {model_choice} predictions..."):
                    predictions = predict_navigation_on_sequences(X_seq, nav_model, model_choice, cnn_extractor)

                pred_classes = np.argmax(predictions, axis=1)
                pred_confidences = np.max(predictions, axis=1) * 100

                st.markdown("---")
                st.subheader("📊 Prediction Results")

                # Summary statistics
                unique, counts = np.unique(pred_classes, return_counts=True)
                col1, col2, col3, col4 = st.columns(4)
                cols = [col1, col2, col3, col4]
                for i, name in enumerate(CLASS_NAMES):
                    with cols[i]:
                        count = counts[unique == i][0] if i in unique else 0
                        pct = (count / len(pred_classes)) * 100
                        st.metric(
                            f"{DIRECTION_EMOJIS[name]} {name}",
                            f"{count}",
                            f"{pct:.1f}%"
                        )

                # Distribution chart
                dist_df = pd.DataFrame({
                    'Direction': [CLASS_NAMES[u] for u in unique],
                    'Count': counts,
                })
                fig = px.pie(dist_df, values='Count', names='Direction',
                             title='Direction Distribution',
                             color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)

                # Frame-by-frame predictions
                with st.expander("🔍 Frame-by-Frame Predictions"):
                    results_df = pd.DataFrame({
                        'Sequence': range(1, len(pred_classes) + 1),
                        'Direction': [CLASS_NAMES[c] for c in pred_classes],
                        'Confidence (%)': [f"{c:.1f}" for c in pred_confidences],
                    })
                    st.dataframe(results_df, use_container_width=True, height=400)

                # Show sample frames with predictions
                st.subheader("🖼️ Sample Predictions")
                n_samples = min(5, len(X_seq))
                sample_cols = st.columns(n_samples)
                step = max(1, len(X_seq) // n_samples)
                for i in range(n_samples):
                    idx = i * step
                    with sample_cols[i]:
                        frame_rgb = (X_seq[idx, -1] * 255).astype(np.uint8)
                        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"{DIRECTION_EMOJIS[CLASS_NAMES[pred_classes[idx]]]} {CLASS_NAMES[pred_classes[idx]]} ({pred_confidences[idx]:.1f}%)", use_container_width=True)


# =====================================================
# SECTION 2: Drone Info Analyzer
# =====================================================

elif section == "📷 Drone Info Analyzer (Image)":
    st.header("📷 Drone Info Analyzer — Image → Type + Speed + Price")
    st.markdown("Upload a drone image to predict its **Type**, **Speed Category**, and **Price Range**.")

    drone_model = load_drone_model()

    if drone_model is None:
        st.error("⚠️ model_drone_info.h5 not found! Please train and save the drone model first.")
    else:
        uploaded_image = st.file_uploader("📤 Upload a drone image", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            col_img, col_results = st.columns([1, 1])

            with col_img:
                st.image(image, caption="Uploaded Drone Image", use_container_width=True)

            with col_results:
                with st.spinner("🔍 Analyzing drone..."):
                    result = predict_drone_info(image, drone_model)

                st.markdown("### 🚁 Analysis Results")

                # Type
                st.markdown(f"""
                <div class="result-card">
                    <h3>🏷️ Type</h3>
                    <h2>{result['Type']}</h2>
                    <p>Confidence: {result['Type Confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # Speed
                speed_emoji = {'High': '🚀', 'Medium': '✈️', 'Low': '🐌'}.get(result['Speed'], '❓')
                st.markdown(f"""
                <div class="result-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3>{speed_emoji} Speed Category</h3>
                    <h2>{result['Speed']}</h2>
                    <p>Confidence: {result['Speed Confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # Price
                st.markdown(f"""
                <div class="result-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h3>💰 Price Range</h3>
                    <h2>{result['Price']}</h2>
                    <p>Confidence: {result['Price Confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # Show confidence bars
            st.markdown("---")
            st.subheader("📊 Detailed Confidence Scores")

            # Rerun full prediction for detailed scores
            img_arr = img_to_array(image.convert('RGB').resize(DRONE_IMG_SIZE)) / 255.0
            img_batch = np.expand_dims(img_arr, axis=0)
            type_pred, speed_pred, price_pred = drone_model.predict(img_batch, verbose=0)

            tab1, tab2, tab3 = st.tabs(["🏷️ Type", "🚀 Speed", "💰 Price"])

            with tab1:
                type_df = pd.DataFrame({
                    'Class': TYPE_CLASSES,
                    'Confidence (%)': (type_pred[0] * 100).round(2)
                }).sort_values('Confidence (%)', ascending=True)
                fig = px.bar(type_df, x='Confidence (%)', y='Class', orientation='h',
                             color='Confidence (%)', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                speed_df = pd.DataFrame({
                    'Class': SPEED_CLASSES,
                    'Confidence (%)': (speed_pred[0] * 100).round(2)
                }).sort_values('Confidence (%)', ascending=True)
                fig = px.bar(speed_df, x='Confidence (%)', y='Class', orientation='h',
                             color='Confidence (%)', color_continuous_scale='Magma')
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                price_df = pd.DataFrame({
                    'Class': PRICE_CLASSES,
                    'Confidence (%)': (price_pred[0] * 100).round(2)
                }).sort_values('Confidence (%)', ascending=True)
                fig = px.bar(price_df, x='Confidence (%)', y='Class', orientation='h',
                             color='Confidence (%)', color_continuous_scale='Cividis')
                st.plotly_chart(fig, use_container_width=True)


# =====================================================
# SECTION 3: Models Comparison
# =====================================================

elif section == "📊 Models Comparison":
    st.header("📊 Navigation Models Comparison")
    st.markdown("Compare all 5 navigation models on the test set.")

    cnn_extractor = load_feature_extractor()

    # Check for test data
    has_test_data = os.path.exists("X.npy") and os.path.exists("y.npy")
    has_features = os.path.exists("X_train_feat.npy") and os.path.exists("X_test_feat.npy")

    if not has_test_data:
        st.warning("⚠️ Test data (X.npy, y.npy) not found. Please run the notebook first.")
        st.info("Showing model file availability only:")
        for name, path in MODEL_FILES.items():
            if os.path.exists(path):
                st.success(f"✅ {name} — `{path}`")
            else:
                st.error(f"❌ {name} — `{path}` missing")
    else:
        if st.button("🔄 Run Full Comparison", type="primary"):
            # Load data exactly like notebook
            X = np.load("X.npy")
            y = np.load("y.npy")

            from tensorflow.keras.utils import to_categorical
            from sklearn.model_selection import train_test_split

            num_classes = 4
            y_cat = to_categorical(y, num_classes=num_classes)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_cat, test_size=0.2, random_state=42, stratify=y
            )

            # Extract features if not cached
            if has_features:
                X_train_feat = np.load("X_train_feat.npy")
                X_test_feat = np.load("X_test_feat.npy")
            else:
                with st.spinner("🔧 Extracting CNN features (this may take a while)..."):
                    X_train_feat = extract_features(X_train, cnn_extractor)
                    X_test_feat = extract_features(X_test, cnn_extractor)

            # Evaluate each model
            results = []
            progress_bar = st.progress(0)

            for idx, (name, path) in enumerate(MODEL_FILES.items()):
                if not os.path.exists(path):
                    st.warning(f"⚠️ {name} ({path}) not found, skipping...")
                    continue

                model = load_nav_model(name)

                if name == 'CNN (Baseline)':
                    X_eval = X_test[:, -1]
                else:
                    X_eval = X_test_feat

                loss, acc = model.evaluate(X_eval, y_test, verbose=0)

                # Get predictions for classification report
                preds = model.predict(X_eval, verbose=0)
                pred_classes = np.argmax(preds, axis=1)
                true_classes = np.argmax(y_test, axis=1)

                from sklearn.metrics import classification_report, confusion_matrix
                report = classification_report(true_classes, pred_classes,
                                               target_names=CLASS_NAMES, output_dict=True)

                results.append({
                    'Model': name,
                    'Accuracy': acc,
                    'Accuracy %': round(acc * 100, 2),
                    'Precision (macro)': round(report['macro avg']['precision'] * 100, 2),
                    'Recall (macro)': round(report['macro avg']['recall'] * 100, 2),
                    'F1 (macro)': round(report['macro avg']['f1-score'] * 100, 2),
                    'pred_classes': pred_classes,
                    'true_classes': true_classes,
                })

                progress_bar.progress((idx + 1) / len(MODEL_FILES))

            progress_bar.empty()

            if not results:
                st.error("No models found! Train and save models first.")
            else:
                df_results = pd.DataFrame(results).sort_values('Accuracy', ascending=False).reset_index(drop=True)

                # Best model highlight
                best = df_results.iloc[0]
                st.markdown(f"""
                <div class="result-card">
                    <h3>🏆 Best Model</h3>
                    <h2>{best['Model']}</h2>
                    <p>Accuracy: {best['Accuracy %']}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Comparison table
                st.subheader("📋 Results Table")
                display_df = df_results[['Model', 'Accuracy %', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)']].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Bar chart
                st.subheader("📊 Accuracy Comparison")
                colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_results['Model'],
                        y=df_results['Accuracy %'],
                        marker_color=colors[:len(df_results)],
                        text=[f"{v}%" for v in df_results['Accuracy %']],
                        textposition='outside',
                    )
                ])
                fig.update_layout(
                    title='Navigation Model Comparison',
                    yaxis_title='Accuracy (%)',
                    yaxis_range=[0, 105],
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Radar chart
                st.subheader("🕸️ Multi-Metric Radar Chart")
                categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                fig_radar = go.Figure()
                for _, row in df_results.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row['Accuracy %'], row['Precision (macro)'],
                           row['Recall (macro)'], row['F1 (macro)']],
                        theta=categories,
                        fill='toself',
                        name=row['Model'],
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    height=500,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # Confusion matrices
                st.subheader("🔥 Confusion Matrices")
                import plotly.figure_factory as ff

                tabs = st.tabs([r['Model'] for r in results])
                for tab, r in zip(tabs, results):
                    with tab:
                        cm = confusion_matrix(r['true_classes'], r['pred_classes'])
                        fig_cm = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="True", color="Count"),
                            x=CLASS_NAMES,
                            y=CLASS_NAMES,
                            text_auto=True,
                            color_continuous_scale='Blues',
                            title=f"{r['Model']} — Confusion Matrix",
                        )
                        fig_cm.update_layout(height=450)
                        st.plotly_chart(fig_cm, use_container_width=True)

                        # Per-class metrics
                        report = classification_report(
                            r['true_classes'], r['pred_classes'],
                            target_names=CLASS_NAMES, output_dict=True
                        )
                        per_class_df = pd.DataFrame({
                            'Class': CLASS_NAMES,
                            'Precision': [round(report[c]['precision'] * 100, 2) for c in CLASS_NAMES],
                            'Recall': [round(report[c]['recall'] * 100, 2) for c in CLASS_NAMES],
                            'F1-Score': [round(report[c]['f1-score'] * 100, 2) for c in CLASS_NAMES],
                            'Support': [report[c]['support'] for c in CLASS_NAMES],
                        })
                        st.dataframe(per_class_df, use_container_width=True, hide_index=True)


# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "🚁 Autonomous Navigation Drone — Multimodel System for Sequence Learning Navigation + Drone Info Analyzer"
    "</p>",
    unsafe_allow_html=True,
)
