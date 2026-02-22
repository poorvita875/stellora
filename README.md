# 🔭 Stellora - Where Space Meets Insight

> NASA publishes real-time data on every asteroid approaching Earth. IRSA hosts millions of telescope images. None of it is readable by normal people. **Stellora fixes that.**

** Live:** [stellora.netlify.app](https://stellora.netlify.app) &nbsp;|&nbsp; ** AI Demo:** [huggingface.co/spaces/Poorvita/astroai](https://huggingface.co/spaces/Poorvita/astroai)

---

## What It Does

Two AI models turn complex astronomical data into instant human-readable intelligence:

| Model | Task | Accuracy |
|-------|------|----------|
|  ResNet50 CNN | Telescope image → Galaxy / Star / Nebula / Planet | 79.3% |
|  Random Forest | Orbital parameters → Hazard risk score | 75.5% |
|  NASA NeoWs API | Live asteroid feed → Real-time radar | Live |

---

## Features

- **Object Classifier** — Upload any telescope image, get instant classification with confidence scores
- **Asteroid Risk Explorer** — Search any asteroid, NASA fetches orbital data, RF scores the threat
- **Live Radar** — Animated real-time radar of all near-Earth objects today
- **3D Visualization** — Three.js rotating planet + asteroid, mouse-reactive


---

## The Real Story

This wasn't smooth. Here's what actually happened:

- **22.5% accuracy** on first training run — dataset was full of mislabeled Google Search images
- Built a custom cleaning pipeline to remove diagrams, watermarks, screenshots
- Switched from EfficientNetB0 → **ResNet50** for better handling of noisy data
- Two-phase training: frozen base → full fine-tune on Kaggle T4 GPU
- Final result: **79.3% CNN · 75.5% RF** — deployed same day

---

## Stack

**Frontend** — HTML, CSS, JS, Three.js, Canvas API, GitHub Pages, ML models and algorithm like RF, ResNet50 etc 
**AI Backend** — TensorFlow, Keras, Scikit-learn, Gradio, Hugging Face Spaces  
**Data** — NASA NeoWs API, NASA Asteroids Dataset, Astronomy Image Dataset (Kaggle)

---

## Architecture

```
User → Stellora (Netlify)
           ├── NASA NeoWs API   →  Live asteroid orbital data
           └── Hugging Face     →  CNN + RF predictions
```

---

## Model Performance

```
              precision    recall    f1
galaxy           0.91      0.67     0.78
star             0.66      0.74     0.70
nebula           0.76      0.82     0.79
planet           0.85      0.97     0.90
overall                             0.79
```

---

## Run Locally

```bash
git clone https://github.com/poorvita875/stellora
# Open index.html in browser — no build step needed
```

---

## Limitations & Roadmap

- Galaxy recall (67%) limited by noisy training data
- NASA key visible in client-side JS — proxy backend planned for v2


---

## Author

**Poorvita** - built by someone who genuinely loves astronomy, astrophysics, and everything about space.

*I visited real NASA databases, IRSA, and NeoWs because I was curious — and found that the data 
was incredible but completely inaccessible. Stellora exists to change that — to make the universe 
readable for everyone who looks up and wonders.*
[![Live](https://img.shields.io/badge/🌐-stellora.netlify.app-blue)](https://stellora.netlify.app)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/Poorvita)
[![GitHub](https://img.shields.io/badge/GitHub-Poorvita-black)](https://github.com/Poorvita)

---

*© 2026 Stellora by Poorvita. All Rights Reserved.*
