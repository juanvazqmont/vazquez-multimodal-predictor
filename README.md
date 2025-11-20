# Multimodal Target Predictor - Deployment

This is a Streamlit app for making predictions using a trained multimodal model that combines tabular data and image features.

## Setup Instructions

# Multimodal Prediction App

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Ensure models are in the `models/` folder
3. Run: `streamlit run app.py`

## Usage
1. Choose CSV upload or manual entry
2. Fill in all tabular features
3. Upload corresponding image
4. Click "Generate Prediction"

## Requirements
- Python 3.8+
- PyTorch
- Streamlit
- CatBoost
- scikit-learn
- Pillow
- pandas, numpy
