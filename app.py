import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Model and parameters
IMG_SIZE = (224, 224)
model = None
class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model, class_names
    
    try:
        # Load model
        model = keras.models.load_model('trained_cow_model.keras')
        print("✅ Model loaded successfully")
        
        # Try to load class names from training data
        try:
            import glob
            data_folder = 'data'
            if os.path.exists(data_folder):
                class_names = sorted([d for d in os.listdir(data_folder) 
                                    if os.path.isdir(os.path.join(data_folder, d))])
                print(f"✅ Loaded class names: {class_names}")
            else:
                # Default classes if data folder doesn't exist
                class_names = ['lumpy', 'abscess', 'culitions']
                print(f"⚠️ Using default class names: {class_names}")
        except:
            class_names = ['lumpy', 'abscess', 'culitions']
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def predict_image(image_path):
    """Make prediction on a single image"""
    if model is None:
        return None
    
    try:
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get results
        results = []
        for i, class_name in enumerate(class_names):
            confidence = float(predictions[i] * 100)
            results.append({
                'class': class_name,
                'confidence': confidence,
                'bar_width': min(confidence, 100)  # For progress bar
            })
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'top_prediction': results[0],
            'all_predictions': results,
            'image_path': image_path
        }
        
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

def create_confidence_chart(predictions):
    """Create a bar chart of predictions"""
    plt.figure(figsize=(8, 4))
    
    classes = [p['class'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']
    
    bars = plt.barh(classes, confidences, color=colors[:len(classes)])
    plt.xlabel('Confidence (%)')
    plt.title('Disease Prediction Confidence')
    plt.xlim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Encode to base64 for HTML
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create uploads folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file
        file.save(filepath)
        
        # Make prediction
        result = predict_image(filepath)
        
        if result:
            # Create confidence chart
            chart_image = create_confidence_chart(result['all_predictions'])
            
            return render_template('result.html', 
                                 result=result,
                                 chart_image=chart_image,
                                 filename=filename)
        else:
            flash('Error making prediction. Please try another image.', 'error')
            return redirect(url_for('index'))
    
    else:
        flash('Allowed file types are: png, jpg, jpeg, gif, bmp', 'error')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html') if os.path.exists('templates/about.html') else "About page"

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Save temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)
        
        # Make prediction
        result = predict_image(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        if result:
            return {
                'success': True,
                'predictions': result['all_predictions'],
                'top_prediction': result['top_prediction']
            }
    
    return {'error': 'Invalid file'}, 400

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Cow Disease Detection App...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🔢 Image size: {IMG_SIZE}")
    print(f"📊 Classes: {class_names}")
    print(f"🌐 Server running at: http://127.0.0.1:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)