from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime
from functools import wraps
import numpy as np
import base64

# ==========================================
# INITIALIZE FLASK APP
# ==========================================
app = Flask(__name__, 
    template_folder='templates',
    static_folder='static')

try:
    model = YOLO('best.pt')  # Load custom trained model
    print("✓ Model siap digunakan 🚀")
except Exception as e:
    print(f"✗ Error model YOLO : {e}")
    model = None

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


  

def detect_objects(image_path):
    """
    Perform YOLO detection on image
    Returns: (annotated_image, detections_list)
    """
    if model is None:
        return None, []
    
    try:
        # Read image
        img = cv2.imread(image_path)
        
        # Run inference
        results = model(img)
        
        # Parse results
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = r.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'{class_name} {conf:.2f}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return img, detections
    
    except Exception as e:
        print(f"Error in detection: {e}")
        return None, []



@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    """Detection page"""
    return render_template('deteksi.html')

@app.route('/tentang')
def tentang():
    """About page"""
    return render_template('tentang.html')

# ==========================================
# DETECTION ENDPOINTS
# ==========================================

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """
    API endpoint for image detection
    Expects: POST with 'image' file and optional 'confidence' parameter
    Returns: JSON with detection results and annotated image
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        confidence = float(request.form.get('confidence', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file to disk
        filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run detection
        annotated_img, detections = detect_objects(filepath)

        if annotated_img is None:
            return jsonify({'error': 'Detection failed'}), 500
        
        # Filter by confidence
        filtered_detections = [d for d in detections if d['confidence'] >= confidence]

        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            'success': True,
            'detections': filtered_detections,
            'detection_count': len(filtered_detections),
            'annotated_image': f'data:image/jpeg;base64,{img_base64}',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Get prediction history from memory (disabled)"""
    return jsonify({'error': 'History feature is disabled. No database integration.'}), 404

# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Server error'}), 500

@app.context_processor
def inject_user():
    """Inject user data into templates"""
    return {
        'user_id': session.get('user_id'),
        'username': session.get('username')
    }

if __name__ == '__main__':
    print("=" * 50)
    print("Vehicle Detection Dashboard")
    print("=" * 50)
    print("🚀 Starting Flask server...")
    print("=" * 50)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)