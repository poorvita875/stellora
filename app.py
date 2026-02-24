import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import joblib
import json
import os

# ── LOAD MODELS ─────────────────────────────────────────────
print("Initializing AstroAI Intelligence System...")
try:
    # Loading CNN with modern Keras compatibility
    cnn_model = load_model('astronomical_classifier.h5', compile=False)
    
    # Load and map class indices
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}
    
    # Load Random Forest for Asteroid Risk
    rf_model = joblib.load('asteroid_risk_rf.pkl')
    rf_features = joblib.load('rf_features.pkl')
    print("All models loaded and active!")
except Exception as e:
    print(f"Initialization Error: {e}")

# ── CLASS INFO ───────────────────────────────────────────────
CLASS_INFO = {
    'galaxy': {'icon': '🌌', 'description': 'A system of billions of stars, gas, and dust.', 'fact': 'There are trillions in the universe.'},
    'star': {'icon': '⭐', 'description': 'A luminous plasma sphere.', 'fact': 'The Sun converts 4M tonnes of matter to energy every second.'},
    'nebula': {'icon': '🌠', 'description': 'An interstellar cloud of gas and dust.', 'fact': 'The Crab Nebula is a supernova remnant.'},
    'planet': {'icon': '🪐', 'description': 'A body orbiting a star.', 'fact': 'Jupiter is more massive than all other planets combined.'}
}

# ── PREDICTION LOGIC ─────────────────────────────────────────
def classify_image(img):
    if img is None: return "Please upload an image.", {}
    
    # Standard ResNet/CNN Preprocessing
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized) / 255.0
    if arr.shape[-1] == 4: arr = arr[:, :, :3]
    arr = np.expand_dims(arr, axis=0)
    
    predictions = cnn_model.predict(arr)[0]
    pred_idx = int(np.argmax(predictions))
    class_raw = idx_to_class.get(pred_idx, "Unknown")
    pred_class = class_raw.lower()
    
    confidence = float(predictions[pred_idx])
    info = CLASS_INFO.get(pred_class, {})
    
    result = f"## {info.get('icon','🔭')} {pred_class.upper()}\n"
    result += f"**Confidence:** {confidence:.1%}\n"
    result += f"**About:** {info.get('description','No info available.')}\n"
    result += f"**Fact:** {info.get('fact','N/A')}"
    
    scores = {}
    for i, prob in enumerate(predictions):
        name = idx_to_class.get(i, f"Class {i}")
        icon = CLASS_INFO.get(name.lower(), {}).get('icon', '🔭')
        scores[f"{icon} {name}"] = float(prob)
        
    return result, scores

def predict_risk(diameter_min, diameter_max, velocity, miss_distance, magnitude):
    # Matches feature order in rf_features.pkl
    input_data = np.array([[diameter_min, diameter_max, velocity, miss_distance, magnitude]])
    pred = rf_model.predict(input_data)[0]
    prob = rf_model.predict_proba(input_data)[0]
    risk_pct = float(prob[1]) * 100
    
    status = "⚠️ POTENTIALLY HAZARDOUS" if pred == 1 else "NOT HAZARDOUS"
    return f"## {status}\n**Hazard Probability:** {risk_pct:.1f}%\n**Safe Probability:** {100-risk_pct:.1f}%"

# ── GRADIO UI ────────────────────────────────────────────────
with gr.Blocks(title="AstroAI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🔭 AstroAI — Astronomical Intelligence System
> 🌐 **For the full experience with 3D visualizations, live asteroid radar and interactive UI:**
> ### [✨ Visit Stellora → stellora.netlify.app](https://stellora.netlify.app)
""")
    
    with gr.Tabs():
        # TAB 1: OBJECT CLASSIFIER
        with gr.Tab("Object Classifier"):
            gr.Markdown("""
            ### 🔭 Classification Scope
            This AI is specifically trained to identify:
            * **Galaxies** (Spiral, Elliptical)
            * **Stars** (Point-source luminaries)
            * **Nebulae** (Interstellar clouds)
            * **Planets** (Spherical celestial bodies)
            * *Note: Other objects may yield inaccurate results.*
            ---
            ### Model Performance Diagnostics
            * **Overall CNN Accuracy:** **79.31%**
            * **Reliability Note:** The model is exceptionally strong at identifying **Planets (97% Recall)**. 
            * **Known Limitations:** It has a **67% Recall for Galaxies**, meaning it can occasionally miss a galaxy or misidentify it if the image is faint or low-resolution.
            """)
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Astronomical Image")
                    classify_btn = gr.Button("🔍 Classify Object", variant="primary")
                with gr.Column():
                    result_md = gr.Markdown(label="Result")
                    confidence_label = gr.Label(label="Confidence Scores", num_top_classes=4)
            
            classify_btn.click(
                fn=classify_image,
                inputs=[image_input],
                outputs=[result_md, confidence_label]
            )

        # TAB 2: ASTEROID RISK
        with gr.Tab("☄️ Asteroid Risk Predictor"):
            gr.Markdown("""
            ### Risk Assessment Engine
            * **Random Forest Accuracy:** **75.5%**
            * Enter orbital parameters manually to assess if a Near-Earth Object (NEO) poses a threat.
            """)
            
            with gr.Row():
                with gr.Column():
                    dmin = gr.Slider(0.001, 50, value=0.5, label="Min Diameter (km)")
                    dmax = gr.Slider(0.001, 100, value=1.0, label="Max Diameter (km)")
                    vel = gr.Slider(1000, 100000, value=25000, label="Relative Velocity (km/h)")
                    miss = gr.Slider(100000, 70000000, value=5000000, label="Miss Distance (km)")
                    mag = gr.Slider(10, 35, value=22, label="Absolute Magnitude (H)")
                    risk_btn = gr.Button("⚠️ Assess Risk", variant="primary")
                with gr.Column():
                    risk_result = gr.Markdown(label="Risk Assessment Result")
            
            risk_btn.click(
                fn=predict_risk,
                inputs=[dmin, dmax, vel, miss, mag],
                outputs=[risk_result]
            )

    gr.Markdown("--- *Project developed for AstroAI monitoring. Metrics verified against NASA and Kaggle Astronomy datasets.*")

if __name__ == "__main__":
    # We remove SSR and other experimental features to stay under memory limits
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True
    )
