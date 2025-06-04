import streamlit as st
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from PIL import Image
import io
import time

# === Configuration ===
IMG_SIZE = (128, 128)
MODEL_PATH = "D:/fire_detection/Fire_Dataset_PCD_Copy/fire_svm_model.pkl"
ENCODER_PATH = "D:/fire_detection/Fire_Dataset_PCD_Copy/label_encoder.pkl"

# === Streamlit Page Config ===
st.set_page_config(
    page_title="🔥 Fire Detection System",
    page_icon="🔥",
    layout="centered"
)

# === Feature Extraction Function (Optimized) ===
@st.cache_data
def extract_color_histogram(image_array):
    """Extract color histogram features from image"""
    image = cv2.resize(image_array, IMG_SIZE)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# === Load Dataset Function ===
@st.cache_data
def load_dataset(directory):
    """Load and process dataset for training"""
    data = []
    labels = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = sum([len(files) for r, d, files in os.walk(directory)])
    processed = 0
    
    for label_name in os.listdir(directory):
        folder_path = os.path.join(directory, label_name)
        if not os.path.isdir(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            if image is None:
                continue
                
            features = extract_color_histogram(image)
            data.append(features)
            labels.append(label_name)
            
            processed += 1
            progress = processed / total_files
            progress_bar.progress(progress)
            status_text.text(f'Processing: {processed}/{total_files} images')
    
    progress_bar.empty()
    status_text.empty()
    return np.array(data), np.array(labels)

# === Train Model Function ===
@st.cache_resource
def train_model(train_dir, test_dir):
    """Train the SVM model"""
    # Load training data
    X_train, y_train = load_dataset(train_dir)
    
    # Load test data  
    X_test, y_test = load_dataset(test_dir)
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Train model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train_enc)
    
    # Save model and encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    
    return model, le, X_test, y_test_enc

# === Load Pre-trained Model ===
@st.cache_resource
def load_model():
    """Load pre-trained model and encoder"""
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return model, le
    except FileNotFoundError:
        return None, None

# === Predict Function ===
def predict_fire(image_array, model, label_encoder):
    """Predict if image contains fire"""
    features = extract_color_histogram(image_array)
    features = features.reshape(1, -1)
    
    # Get prediction and probability
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get class name and confidence
    class_name = label_encoder.inverse_transform([prediction])[0]
    confidence = np.max(probabilities)
    
    return class_name, confidence

# === Main App ===
def main():
    st.title("🔥 Fire Detection System")
    st.markdown("Upload an image to detect if it contains fire or not!")
    
    # Sidebar for model training
    st.sidebar.header("🛠️ Model Training")
    
    train_dir = st.sidebar.text_input("Training Directory Path:", 
                                     value="D:/fire_detection/Fire_Dataset_PCD_Copy/Train")
    test_dir = st.sidebar.text_input("Test Directory Path:", 
                                    value="D:/fire_detection/Fire_Dataset_PCD_Copy/Test")
    
    if st.sidebar.button("🚀 Train New Model"):
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    model, le, X_test, y_test_enc = train_model(train_dir, test_dir)
                    st.sidebar.success("✅ Model trained successfully!")
                    
                    # Show training results
                    y_pred = model.predict(X_test)
                    accuracy = np.mean(y_test_enc == y_pred)
                    st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Training failed: {str(e)}")
        else:
            st.sidebar.error("❌ Please check directory paths!")
    
    # Load existing model
    model, le = load_model()
    
    if model is None:
        st.warning("⚠️ No trained model found. Please train a model first using the sidebar.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to check for fire detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            # Convert PIL to OpenCV format
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                prediction, confidence = predict_fire(image_array, model, le)
                processing_time = time.time() - start_time
            
            # Display results
            if prediction.lower() == 'fire':
                st.error(f"🔥 **FIRE DETECTED!**")
                st.error(f"Confidence: {confidence:.2%}")
            else:
                st.success(f"✅ **NO FIRE DETECTED**")
                st.success(f"Confidence: {confidence:.2%}")
            
            st.info(f"⚡ Processing time: {processing_time:.3f} seconds")
            
            # Confidence meter
            st.subheader("📊 Confidence Level")
            st.progress(confidence)
            st.caption(f"Model is {confidence:.1%} confident in this prediction")
    
    # Instructions
    with st.expander("📋 How to use"):
        st.markdown("""
        1. **First time setup**: Use the sidebar to train a new model with your dataset
        2. **Upload**: Click 'Browse files' to upload an image
        3. **Wait**: The system will analyze the image for fire detection
        4. **Results**: View the prediction with confidence score
        
        **Supported formats**: PNG, JPG, JPEG
        **Processing time**: Usually under 1 second per image
        """)
    
    # Performance tips
    with st.expander("⚡ Performance Tips"):
        st.markdown("""
        - **Smaller images** process faster (images are resized to 128x128 automatically)
        - **Good lighting** improves accuracy  
        - **Clear fire/smoke** is easier to detect than subtle flames
        - **Multiple angles** of the same scene can improve confidence
        """)

if __name__ == "__main__":
    main()