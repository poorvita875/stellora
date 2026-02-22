
##Stellora — Where Space Meets Insight
AI-powered astronomical classifier with live NASA asteroid radar

Turning raw NASA data into something every human can understand.

AI Demo: huggingface.co/spaces/Poorvita/astroai
Built by: Poorvita | © 2026 All Rights Reserved

# The Problem That Started This :
Have you ever visited NASA's NeoWs API, IRSA, or any real astronomical database?


They look like this:


{
  "absolute_magnitude_h": 22.1,
  "estimated_diameter": {
    "kilometers": {
      "estimated_diameter_min": 0.0531,
      "estimated_diameter_max": 0.1188
    }
  },
  "close_approach_data": [{
    "relative_velocity": { "kilometers_per_hour": "42589.1674527605" },
    "miss_distance": { "kilometers": "4823918.304590" }
  }]
}


Raw. Cold. Incomprehensible to anyone who isn't a professional astronomer.
I visited these websites myself. I spent hours trying to understand what I was looking at - pages of orbital parameters, FITS image files that require special software to open, magnitude numbers with no context, miss distances in kilometers that mean nothing without a reference point.
NASA and research institutions produce extraordinary scientific data. But their interfaces are built for experts - not for students, not for curious minds, not even for researchers from adjacent fields who just need a quick answer.


The question I asked myself: What if anyone - a student, a curious person, a researcher - could understand all of this in seconds? Visually. Intelligently. Beautifully?
That's what Stellora is.

 Stellora bridges the gap between complex astronomical data and 
human understanding. Upload a telescope image → ResNet50 CNN 
classifies it instantly. Search any asteroid → NASA orbital data 
+ Random Forest returns a plain-language risk score in seconds.

  
# The Real Journey - What Actually Happened
This project was not smooth. Here is the complete honest story of everything that was built, broken, debugged, and learned.
The Dataset Problem (Week 1)
The first training attempt used a Google Search-scraped astronomy image dataset with 758 images across 4 classes. After full training, the CNN achieved 22.5% accuracy - essentially random guessing.
Diagnosis: The dataset was fundamentally broken.

Planet folder contained artistic renders, solar system diagrams with text labels like "Jupiter Mars Venus Moon"
Galaxy folder contained planet renders and artistic illustrations mixed in
Star folder had Milky Way landscape photos and star trails — not actual stellar objects
Images had watermarks, logos, borders, and low resolution

No model architecture can compensate for this. The CNN was trying to find features that didn't exist consistently.
Data Cleaning Pipeline
Built an automated image quality filter that detected and removed:

Images below 100×100 pixels (thumbnails)
Images with aspect ratio > 2.5 (diagrams and banners)
Images with pixel standard deviation < 15 (solid color placeholders)
Images with mean brightness > 230 (web page screenshots)
Corrupt and unreadable files

Hardware: Kaggle T4 GPU, ~60 minutes total training time

Final Model Results :
🧠 FINAL CNN ACCURACY: 79.31%

              precision    recall  f1-score   support
      galaxy       0.91      0.67      0.78        46
        star       0.66      0.74      0.70        31
      nebula       0.76      0.82      0.79        34
      planet       0.85      0.97      0.90        34

    accuracy                           0.79       145
Planet at 97% recall on a real-world noisy dataset is a genuinely strong result. Galaxy at 67% recall reflects the underlying data quality issue — a limitation honestly documented.
Random Forest for Asteroid Risk
Trained on the NASA Asteroids Classification Dataset (Kaggle):

Final accuracy: 75.5%

Three-Tier Deployment
Designed a complete production architecture connecting three free platforms:
Kaggle (training) → Hugging Face (serving) → GitHub Pages (frontend)
No paid infrastructure. Fully deployed. Fully functional.

✨ Features :

1) CNN Object Classifier — Upload any telescope image, ResNet50 identifies it with animated confidence bars
2) Asteroid Risk Explorer — Search any asteroid by name, NASA + RF delivers instant risk assessment
3) Live Asteroid Radar — Animated real-time radar of all near-Earth objects today, yesterday, tomorrow
4) 3D Visualization — Three.js rotating planet and asteroid with ring system, mouse-reactive
5) Particle Nebula — 120-particle nebula background that reacts to mouse movement
6) Animated Confidence Bars — Visual confidence scores with smooth reveal animations


# Model Performance :
ResNet50 CNN
ClassPrecisionRecallF1Galaxy0.910.670.78Star0.660.740.70Nebula0.760.820.79Planet0.850.970.90Overall0.810.790.79
Random Forest
ParameterValueAccuracy75.5%Trees200Max Depth15Class WeightBalancedFeatures5 orbital parameters

# Tech Stack
Frontend: HTML5, CSS3, Vanilla JavaScript, Three.js r128, Canvas API, Exo 2 + Space Mono fonts, GitHub Pages
AI Backend: TensorFlow 2.16, Keras, ResNet50, Scikit-learn Random Forest, Gradio, Hugging Face Spaces
Data: NASA NeoWs API, NASA Asteroids Classification Dataset, Astronomy Image Classification Dataset
Training: Kaggle Notebooks, T4 GPU, Python 3.10

# Running Locally
bashgit clone https://github.com/poorvita875/stellora
cd stellora
Open index.html in any browser — no build step required
AI backend locally:
bashpip install tensorflow gradio scikit-learn joblib pillow numpy pandas
python app.py
 Visit http://localhost:7860

📁 Structure
stellora/
├── index.html        # Complete frontend
├── app.py            # Gradio backend (on HF Spaces)
├── requirements.txt  # Python dependencies
├── .gitignore        # Git ignore rules
└── README.md         # This file
Model files (.h5, .pkl) are on Hugging Face due to GitHub's 100MB limit.

### Limitations & Future Work :

Galaxy recall (67%) limited by noisy training dataset
NASA API: 1,000 requests/hour on free tier

Roadmap:

 Galaxy morphology (spiral/elliptical/irregular) using Galaxy Zoo
 FITS file direct upload support
 SDSS spectral classification
 3D asteroid orbit path visualization
 Mobile app


# Acknowledgements

NASA NeoWs API - for making real asteroid data publicly accessible
IRSA - for showing me how complex raw astronomical data really is
Kaggle - for free GPU compute
Hugging Face - for free model hosting
Three.js - for making 3D in the browser possible


## Author
Poorvita - built with curiosity, a lot of failed training runs, and a deep frustration with how inaccessible scientific data is to normal people.
This project exists because I visited real NASA databases and spent hours confused.**Nobody should have to feel that way about the universe.**
© 2026 stellora by Poorvita. All Rights Reserved.
