import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys

# Add the current directory to path
sys.path.append('.')

# Test the model directly
def test_model_directly():
    print("🧪 Testing model directly...")
    
    try:
        # Load model
        model = keras.models.load_model('cow_disease_model.keras')
        print("✅ Model loaded successfully")
        
        # Print model summary
        print("\n📋 Model Summary:")
        model.summary()
        
        # Create a dummy input
        input_shape = model.input_shape[1:]  # Remove batch dimension
        print(f"\n📊 Expected input shape: {input_shape}")
        
        # Create random test image
        test_image = np.random.randn(*input_shape)
        
        # Normalize if needed (based on your training)
        if model.input_shape[-1] == 3:  # RGB image
            # Assuming images were normalized to [0, 1]
            test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
        
        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make prediction
        print("\n🤖 Making prediction on random image...")
        prediction = model.predict(test_image, verbose=0)
        
        print(f"\n📊 Prediction shape: {prediction.shape}")
        print(f"📊 Prediction values: {prediction}")
        
        if prediction.shape[1] > 1:
            print(f"\n🔍 Class probabilities:")
            for i in range(prediction.shape[1]):
                print(f"  Class {i}: {prediction[0][i]:.4f} ({prediction[0][i]*100:.2f}%)")
            
            # Check if predictions are all the same
            if np.allclose(prediction[0], prediction[0][0]):
                print("\n⚠️ WARNING: All class probabilities are nearly equal!")
                print("   This suggests the model may not be trained properly.")
            else:
                print(f"\n✅ Model appears to differentiate between classes.")
                predicted_class = np.argmax(prediction[0])
                print(f"🏆 Predicted class index: {predicted_class}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_with_actual_images():
    print("\n" + "="*50)
    print("🧪 Testing with actual images...")
    
    from app import predict_image, load_model
    
    # Load model through your app's function
    load_model()
    
    # Test with sample images if they exist
    test_images = []
    
    # Look for test images in static/uploads
    if os.path.exists('static/uploads'):
        import glob
        test_images = glob.glob('static/uploads/*.jpg') + glob.glob('static/uploads/*.png') + glob.glob('static/uploads/*.jpeg')
    
    if test_images:
        print(f"\n📸 Found {len(test_images)} test images")
        for i, img_path in enumerate(test_images[:3]):  # Test first 3 images
            print(f"\n{'='*30}")
            print(f"Testing image {i+1}: {os.path.basename(img_path)}")
            result = predict_image(img_path)
            if result:
                print(f"✅ Prediction successful")
                print(f"🏆 Top prediction: {result['top_prediction']['class']} ({result['top_prediction']['confidence']:.2f}%)")
            else:
                print(f"❌ Prediction failed")
    else:
        print("\n📸 No test images found in static/uploads")
        print("Please upload some images through the web interface first.")

if __name__ == "__main__":
    print("🔍 Starting model debugging...")
    model = test_model_directly()
    if model:
        test_with_actual_images()
    print("\n✅ Debugging complete!")