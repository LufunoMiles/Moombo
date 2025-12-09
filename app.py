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
IMG_SIZE = (150, 150)
model = None
class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model, class_names
    
    try:
        # Load model
        model = keras.models.load_model('cow_disease_model.keras')
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
                class_names = ['not a cow', 'abscess', 'lumpy']
                print(f"⚠️ Using default class names: {class_names}")
        except:
            class_names = ['lumpy', 'abscess', 'not a cow']
            
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
        
        print(f"Raw predictions: {predictions}")  # Debug line
        
        # Get results
        results = []
        
        # If predictions is a single value (binary classification)
        if len(predictions.shape) == 0 or len(predictions) == 1:
            # Binary classification
            confidence = float(predictions * 100)
            if len(class_names) >= 2:
                results.append({
                    'class': class_names[1],
                    'confidence': confidence,
                    'bar_width': min(confidence, 100)
                })
                results.append({
                    'class': class_names[0],
                    'confidence': 100 - confidence,
                    'bar_width': min(100 - confidence, 100)
                })
            else:
                # Fallback if class_names not properly loaded
                results.append({
                    'class': 'lumpy' if confidence > 50 else 'not a cow',
                    'confidence': confidence if confidence > 50 else 100 - confidence,
                    'bar_width': min(confidence if confidence > 50 else 100 - confidence, 100)
                })
        else:
            # Multi-class classification
            for i, class_name in enumerate(class_names):
                confidence = float(predictions[i] * 100)
                results.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bar_width': min(confidence, 100)
                })
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Processed results: {results}")  # Debug line
        
        return {
            'top_prediction': results[0],
            'all_predictions': results,
            'image_path': image_path
        }
        
    except Exception as e:
        print(f"Error predicting image: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# Add this function to app.py
def get_disease_info(disease_class, confidence):
    """Get disease information, causes, and treatment suggestions"""
    
    disease_info = {
        'lumpy': {
            'name': 'Lumpy Skin Disease',
            'description': 'A viral disease in cattle caused by the lumpy skin disease virus (LSDV).',
            'symptoms': [
                'Skin nodules (2-5 cm in diameter)',
                'Fever (>40.5°C)',
                'Lacrimation and nasal discharge',
                'Loss of appetite',
                'Reduced milk production',
                'Swollen lymph nodes'
            ],
            'causes': [
                'Virus transmission through insects (mosquitoes, flies)',
                'Direct contact with infected animals',
                'Contaminated equipment or environment',
                'Artificial insemination with infected semen'
            ],
            'treatment': [
                'Isolate infected animals immediately',
                'Symptomatic treatment with anti-inflammatory drugs',
                'Antibiotics to prevent secondary bacterial infections',
                'Good wound care for skin lesions',
                'Supportive care including fluid therapy',
                'Consult veterinarian for antiviral treatment options'
            ],
            'prevention': [
                'Vaccination (Neethling strain vaccine)',
                'Insect control measures',
                'Quarantine new animals for 28 days',
                'Disinfect equipment regularly',
                'Avoid movement of animals from infected areas'
            ],
            'severity': 'High',
            'contagious': True,
            'mortality_rate': '1-5% (can be higher in severe cases)'
        },
        'abscess': {
            'name': 'Abscess',
            'description': 'A localized collection of pus caused by bacterial infection, often from wounds or injections.',
            'symptoms': [
                'Swollen, painful lump',
                'Warm to touch',
                'May rupture and drain pus',
                'Fever in systemic cases',
                'Loss of appetite',
                'Lameness if on legs'
            ],
            'causes': [
                'Bacterial infection (often Staphylococcus or Streptococcus)',
                'Wounds from sharp objects',
                'Poor injection techniques',
                'Foreign bodies (splinters, thorns)',
                'Poor hygiene conditions'
            ],
            'treatment': [
                'Veterinary consultation for proper drainage',
                'Antibiotic therapy (based on culture and sensitivity)',
                'Hot compress to promote maturation',
                'Surgical drainage if large or deep',
                'Wound cleaning with antiseptic solutions',
                'Pain management with anti-inflammatory drugs'
            ],
            'prevention': [
                'Proper injection techniques',
                'Regular wound inspection and treatment',
                'Clean living environment',
                'Remove sharp objects from pens',
                'Good herd hygiene practices'
            ],
            'severity': 'Medium',
            'contagious': False,
            'mortality_rate': 'Very low with proper treatment'
        },
        'not a cow': {
            'name': 'Not a Cow',
            'description': 'The uploaded image does not appear to contain cattle or is not suitable for analysis.',
            'symptoms': [],
            'causes': [
                'Image does not contain cattle',
                'Poor image quality',
                'Wrong animal species',
                'Incorrect camera angle',
                'Image too dark or blurry'
            ],
            'treatment': [
                'Upload a clear photo of cattle',
                'Ensure good lighting conditions',
                'Position camera to show the animal clearly',
                'Focus on specific areas if checking for disease',
                'Take multiple photos from different angles'
            ],
            'prevention': [],
            'severity': 'None',
            'contagious': False,
            'mortality_rate': 'N/A'
        },
        'healthy': {
            'name': 'Healthy Cattle',
            'description': 'No signs of disease detected. The animal appears to be in good health.',
            'symptoms': [],
            'causes': [],
            'treatment': [
                'Continue regular health monitoring',
                'Maintain vaccination schedule',
                'Provide balanced nutrition',
                'Ensure clean water supply',
                'Regular deworming program'
            ],
            'prevention': [
                'Regular veterinary check-ups',
                'Good nutrition and housing',
                'Stress reduction',
                'Biosecurity measures',
                'Regular exercise'
            ],
            'severity': 'None',
            'contagious': False,
            'mortality_rate': 'N/A'
        }
    }
    
    # Default to 'healthy' if not found
    info = disease_info.get(disease_class.lower(), disease_info.get('healthy', {}))
    
    # Add confidence to the info
    info['detection_confidence'] = f"{confidence:.1f}%"
    
    return info

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
            # Get disease information
            disease_info = get_disease_info(
                result['top_prediction']['class'], 
                result['top_prediction']['confidence']
            )
            
            # Create confidence chart
            chart_image = create_confidence_chart(result['all_predictions'])
            
            return render_template('result.html', 
                                 result=result,
                                 disease_info=disease_info,
                                 chart_image=chart_image,
                                 filename=filename)
        else:
            flash('Error making prediction. Please try another image.', 'error')
            return redirect(url_for('index'))
    
    else:
        flash('Allowed file types are: png, jpg, jpeg, gif, bmp', 'error')
        return redirect(url_for('index'))

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