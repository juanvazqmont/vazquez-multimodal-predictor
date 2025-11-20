import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import streamlit as st  # Import here, but don't use yet
warnings.filterwarnings('ignore')

WEIGHTS_DIR = "./model_weights"

def download_model_weights():
    """Manually download and verify model weights"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    resnet_url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    resnet_path = os.path.join(WEIGHTS_DIR, "resnet18-f37072fd.pth")
    
    if not os.path.exists(resnet_path):
        print("Downloading ResNet18 weights...")
        urllib.request.urlretrieve(resnet_url, resnet_path)
        print("✅ ResNet18 weights downloaded")
    
    return resnet_path



class ProductionMultimodalPredictor:
    def __init__(self, model_path='models'):
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(" Using Metal GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(" Using CUDA GPU")
        else:
            self.device = torch.device('cpu')
            print(" Using CPU")
        
        # Download/load model weights first
        self.resnet_weights_path = download_model_weights()
        
        # CNN Ensemble
        self.backbones = ['resnet18']
        self.image_extractors = {}
        self.feature_dims = {}
        
        for backbone in self.backbones:
            extractor, feature_dim = self._build_cnn(backbone)
            self.image_extractors[backbone] = extractor.to(self.device)
            self.feature_dims[backbone] = feature_dim
        
        # CRITICAL: Define image transform here
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Tabular feature metadata (will be loaded from model)
        self.tabular_features = None
        self.seen_image_classes = None
        self.scaler = None
        self.target_scaler = None
        self.catboost_models = []
        
        # Load tabular model
        self.load(model_path)
        

    def _build_cnn(self, backbone_name):
        """Build CNN feature extractor using local weights"""
        import torch.nn as nn
        
        if backbone_name == 'resnet18':
            # Load from local file (bypasses corrupted cache)
            model = models.resnet18(pretrained=False)  # Don't auto-download
            state_dict = torch.load(self.resnet_weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            features = nn.Sequential(*list(model.children())[:-1])
            dim = 512
            
            for param in features.parameters():
                param.requires_grad = False
            
            return features, dim
        
        elif backbone_name == 'efficientnet_b0':
            # SKIP in deployment - causes cache corruption
            st.warning("⚠️ Skipping EfficientNet_b0 due to cache issues. Using ResNet18 only.")
            return None, 0  # Return placeholder
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

    
    def load_image(self, path):
        from PIL import Image
        try:
            if not os.path.exists(path):
                return None
            return self.image_transform(Image.open(path).convert('RGB')).unsqueeze(0)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None
    
    def extract_image_features(self, image_paths, average_ensemble=True):
        """Extract features from all CNNs"""
        from tqdm import tqdm
        
        all_features = {}
        from torchvision import transforms
        from PIL import Image
        
        for backbone_name, extractor in self.image_extractors.items():
            features = []
            with torch.no_grad():
                for path in tqdm(image_paths, desc=f" {backbone_name}", leave=False):
                    tensor = self.load_image(path)
                    if tensor is not None:
                        feat = extractor(tensor.to(self.device))
                        features.append(feat.cpu().numpy().reshape(-1, self.feature_dims[backbone_name]))
                    else:
                        features.append(np.zeros((1, self.feature_dims[backbone_name])))
            
            all_features[backbone_name] = np.vstack(features)
        
        if average_ensemble:
            return np.hstack([all_features[bn] for bn in self.backbones])
        else:
            return all_features
    
    def engineer_features(self, df, fit=False):
        """Advanced feature engineering - Create derived features BEFORE scaling"""
        df = df.copy()
        
        # Handle missing columns
        required_cols = ['yyz', 'qgg', 'lux', 'image']
        optional_cols = ['date', 'bar', 'baz', 'xgt', 'wsg', 'drt', 'gox', 'foo', 'boz', 'fyt', 'lgh', 'hrt', 'juu']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing")
        
        for col in optional_cols:
            if col not in df.columns:
                if col == 'date':
                    df[col] = pd.Timestamp('2020-01-01')
                else:
                    df[col] = 0.0
        
        # ============= STEP 2: Create ALL Derived Features =============
        # Use RAW values for all calculations
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().any():
            df['date'].fillna(pd.Timestamp('2020-01-01'), inplace=True)
        
        df['year'] = df['date'].dt.year
        df['month_raw'] = df['date'].dt.month
        df['dayofweek_raw'] = df['date'].dt.dayofweek
        
        # CYCLICAL (from raw integers)
        df['month'] = df['month_raw'].fillna(6)
        df['dayofweek'] = df['dayofweek_raw'].fillna(3)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # POLYNOMIAL (from raw values) - NEVER SCALE THESE
        df['yyz_squared'] = df['yyz'] ** 2
        df['qgg_squared'] = df['qgg'] ** 2
        df['lux_squared'] = df['lux'] ** 2
        
        # Other features
        df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # ============= STEP 3: Image Class =============
        if 'image class' not in df.columns:
            df['image class'] = df['image'].apply(lambda x: x.split('/')[1] if '/' in x else 'unknown')
        
        if fit:
            self.seen_image_classes = set(df['image class'].unique())
        else:
            df['image class'] = df['image class'].apply(lambda x: x if x in self.seen_image_classes else 'unknown')
        
        # ============= STEP 4: Define Features =============
        if fit and not hasattr(self, 'tabular_features'):
            exclude_cols = ['target', 'image', 'date', 'month_raw', 'dayofweek_raw']
            self.tabular_features = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.tabular_features].copy()
        
        # ============= STEP 5: Selective Scaling =============
        exclude_from_scaling = [
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'yyz_squared', 'qgg_squared', 'lux_squared',
            'is_weekend'
        ]
        
        numeric_cols = []
        for col in X.columns:
            if col in exclude_from_scaling:
                continue
            if X[col].dtype in [np.number]:
                numeric_cols.append(col)
        
        # Apply scaling ONLY to approved columns
        if fit:
            self.scaler = StandardScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        else:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        # Outlier capping (only on scaled features)
        if fit:
            self.outlier_caps = {}
            for col in numeric_cols:
                q1, q99 = X[col].quantile([0.01, 0.99]).values
                self.outlier_caps[col] = (q1, q99)
        
        for col, (q1, q99) in getattr(self, 'outlier_caps', {}).items():
            if col in X.columns:
                X[col] = X[col].clip(q1, q99)
        
        return X.fillna(0)
    
    def _build_feature_matrix(self, tabular_data, image_features):
        """Build final feature matrix"""
        X_tab = tabular_data.values
        
        # Add interactions
        top_features = ['yyz', 'qgg', 'gox', 'lux']
        parts = [X_tab, image_features]
        
        for feat in top_features:
            if feat in tabular_data.columns:
                feat_idx = tabular_data.columns.get_loc(feat)
                feat_col = X_tab[:, feat_idx].reshape(-1, 1)
                interactions = feat_col * image_features
                parts.append(interactions)
        
        return np.hstack(parts)
    
    def predict_ensemble(self, df, image_dir):
        """Ensemble prediction with uncertainty"""
        # Extract features
        image_paths = [os.path.join(image_dir, path) for path in df['image']]
        image_features = self.extract_image_features(image_paths)
        
        tabular_data = self.engineer_features(df, fit=False)
        X = self._build_feature_matrix(tabular_data, image_features)
        
        # Predict with each model
        predictions = []
        for i, model in enumerate(self.catboost_models):
            pred_scaled = model.predict(X)
            pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            predictions.append(pred)
        
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)
    
    def load(self, path='production_models'):
        """Load ensemble with NumPy 1.24 fallback"""
        from catboost import CatBoostRegressor
        
        # Load models
        self.catboost_models = []
        model_files = [f for f in os.listdir(path) if f.endswith('.cbm')]
        for model_file in sorted(model_files):
            model = CatBoostRegressor()
            model.load_model(os.path.join(path, model_file))
            self.catboost_models.append(model)
        
        # Load metadata
        with open(f"{path}/metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.tabular_features = metadata['tabular_features']
            self.seen_image_classes = metadata['seen_image_classes']
            self.outlier_caps = metadata.get('outlier_caps', {})
            self.feature_dim = metadata['feature_dim']
            
            # Target scaler
            self.target_scaler = StandardScaler()
            self.target_scaler.mean_ = np.array([metadata['target_scaler_mean']])
            self.target_scaler.scale_ = np.array([metadata['target_scaler_scale']])
            
            # Feature scaler - handle both NumPy 1.x and 2.x formats
            self.scaler = StandardScaler()
            try:
                # NumPy 1.24 format
                self.scaler.mean_ = np.array(metadata['scaler_mean'])
                self.scaler.scale_ = np.array(metadata['scaler_scale'])
                self.scaler.var_ = np.array(metadata['scaler_var'])
                self.scaler.n_features_in_ = metadata['scaler_n_features_in']
            except KeyError:
                # Fallback: if metadata was corrupted, fit on dummy data
                st.warning("⚠️ Scaler metadata corrupted, refitting on dummy data...")
                dummy_X = np.random.randn(10, len(self.tabular_features))
                self.scaler.fit(dummy_X)
        
        print(f" Loaded ensemble of {len(self.catboost_models)} models")

def main():
    st.set_page_config(page_title="Multimodal Prediction App", page_icon="🎯", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .error-box {
        background-color: #fee;
        color: #c33;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #fcc;
    }
    .success-box {
        background-color: #efe;
        color: #3a3;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #cfc;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🎯 Multimodal Prediction App</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("App Controls")
    st.sidebar.markdown("Configure your prediction settings")
    
    model_path = st.sidebar.text_input("Model Path", value="models", help="Path to trained model directory")
    
    # Initialize predictor
    @st.cache_resource
    def load_predictor(path):
        """Load model with caching"""
        try:
            with st.spinner("Loading model... (this may take 30-60 seconds)"):
                predictor = ProductionMultimodalPredictor(model_path=path)
            st.success("Model loaded successfully!")
            return predictor
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None
    
    predictor = load_predictor(model_path)
    
    if predictor is None:
        st.stop()
    
    # Instructions
    with st.expander("📖 Instructions (Click to expand)", expanded=False):
        st.markdown("""
        ### How to Use This App
        
        **Method 1: CSV Upload (Recommended)**
        1. Upload a CSV file with tabular features
        2. Upload the corresponding image
        3. Click "Predict"
        
        **Method 2: Manual Input**
        1. Fill in all tabular feature fields
        2. Upload an image
        3. Click "Predict"
        
        **Important Notes:**
        - Image must match the tabular data row
        - All numerical features are required
        - Date should be in YYYY-MM-DD format
        - Missing values will be filled with defaults
        """)
    
    # Input method selection
    input_method = st.radio("Choose Input Method", ["Upload CSV", "Manual Entry"])
    
    if input_method == "Upload CSV":
        st.subheader("📁 Upload Tabular Data (CSV)")
        uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_csv is not None:
            try:
                df_input = pd.read_csv(uploaded_csv)
                st.success(f"Loaded {len(df_input)} rows")
                st.dataframe(df_input.head())
            except Exception as e:
                st.error(f" Error loading CSV: {str(e)}")
                st.stop()
        else:
            st.info("Please upload a CSV file")
            st.stop()
    
    else:  # Manual entry
        st.subheader("Manual Tabular Data Entry")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            # Date input
            date_input = st.date_input("Date", value=pd.Timestamp('2020-01-01'))
            
            # Required numerical features
            yyz = st.number_input("yyz", value=0.0, format="%.6f")
            qgg = st.number_input("qgg", value=0.0, format="%.6f")
            lux = st.number_input("lux", value=0.0, format="%.2f")
            bar = st.number_input("bar", value=0.0, format="%.6f")
            baz = st.selectbox("baz", options=[0, 1])
            xgt = st.number_input("xgt", value=0.0, format="%.6f")
            wsg = st.number_input("wsg", value=0.01, format="%.6f")
            
        with col2:
            drt = st.number_input("drt", value=0.0, format="%.6f")
            gox = st.number_input("gox", value=0.0, format="%.6f")
            foo = st.number_input("foo", value=0.0, format="%.6f")
            boz = st.number_input("boz", value=0.0, format="%.6f")
            fyt = st.selectbox("fyt", options=[0, 1])
            lgh = st.selectbox("lgh", options=[0, 1])
            hrt = st.number_input("hrt", value=0.0, format="%.6f")
            juu = st.number_input("juu", value=0.0, format="%.6f")
        
        # Create dataframe from manual input
        df_input = pd.DataFrame([{
            'date': str(date_input),
            'yyz': yyz,
            'qgg': qgg,
            'lux': lux,
            'bar': bar,
            'baz': baz,
            'xgt': xgt,
            'wsg': wsg,
            'drt': drt,
            'gox': gox,
            'foo': foo,
            'boz': boz,
            'fyt': fyt,
            'lgh': lgh,
            'hrt': hrt,
            'juu': juu,
            'image': 'manual_entry.jpg'  # Placeholder
        }])
        
        st.success("Manual data entry complete")
    
    # Image upload
    st.subheader(" Upload Image")
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True, width=300)
        
        # Save temporarily
        temp_image_path = "temp_uploaded_image.jpg"
        image.save(temp_image_path)
        
        # Update image path in dataframe
        df_input['image'] = temp_image_path
        
        st.success("Image uploaded successfully")
    else:
        st.info("Please upload an image")
        st.stop()
    
    # Prediction button
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        try:
            with st.spinner("Running prediction... (this may take a few seconds)"):
                # Run prediction
                predictions, uncertainty = predictor.predict_ensemble(df_input, os.getcwd())
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                pred_value = predictions[0]
                unc_value = uncertainty[0]
                
                # Main prediction display
                st.markdown(f"""
                <div class="prediction-box">
                    Predicted Target: ${pred_value:,.2f}<br>
                    <small>± ${unc_value:,.2f} (uncertainty)</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence indicator
                confidence = "High" if unc_value < pred_value * 0.1 else "Medium" if unc_value < pred_value * 0.2 else "Low"
                confidence_color = "🟢" if confidence == "High" else "🟡" if confidence == "Medium" else "🔴"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", f"${pred_value:,.2f}")
                with col2:
                    st.metric("Uncertainty (1σ)", f"${unc_value:.2f}")
                
                st.info(f"**Confidence Level:** {confidence_color} {confidence}")
                
                # Show details
                with st.expander("🔍 View Detailed Results"):
                    results_df = pd.DataFrame({
                        'Metric': ['Prediction', 'Uncertainty', 'Lower Bound (1σ)', 'Upper Bound (1σ)'],
                        'Value': [
                            f"${pred_value:,.2f}",
                            f"${unc_value:.2f}",
                            f"${pred_value - unc_value:,.2f}",
                            f"${pred_value + unc_value:,.2f}"
                        ]
                    })
                    st.table(results_df)
                
                # Download results
                result_csv = pd.DataFrame({
                    'predicted_target': [pred_value],
                    'uncertainty': [unc_value]
                }).to_csv(index=False)
                
                st.download_button(
                    label="Download Prediction Results (CSV)",
                    data=result_csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)  # Show full traceback in dev mode
    
    # Cleanup temp file
    if os.path.exists("temp_uploaded_image.jpg"):
        os.remove("temp_uploaded_image.jpg")

if __name__ == "__main__":
    main()
