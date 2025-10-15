#!/usr/bin/env python3
"""
MAR EYE CNN API Server
Flask API server for the deployed CNN backend
"""

import os
import sys
import json
import base64
import io
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from model import UNet
from enhanced_inference import EnhancedInference
from run_analytics import run_analytics
from onnx_export import ONNXExporter
from edge_deployment import EdgeDeployment

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path='snapshots/unetSSIM/model_epoch_4_unetSSIM_MODEL.ckpt'):
    """Load the trained CNN model"""
    try:
        model = UNet()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global model instance
cnn_model = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MAR EYE CNN API',
        'model_loaded': cnn_model is not None,
        'version': '1.0.0'
    })

@app.route('/api/process-image', methods=['POST'])
def process_image():
    """Process single image with CNN model"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        if cnn_model is None:
            return jsonify({'success': False, 'error': 'CNN model not loaded'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)
        
        # Process image
        enhanced_inference = EnhancedInference(cnn_model)
        result = enhanced_inference.process_single_image(input_path)
        
        if result['success']:
            # Read enhanced image
            enhanced_path = result['enhanced_path']
            with open(enhanced_path, 'rb') as f:
                enhanced_data = base64.b64encode(f.read()).decode()
            
            return jsonify({
                'success': True,
                'data': {
                    'original_path': input_path,
                    'enhanced_path': enhanced_path,
                    'enhanced_data': enhanced_data,
                    'filename': filename
                },
                'metrics': result['metrics']
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process video with CNN model"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        if cnn_model is None:
            return jsonify({'success': False, 'error': 'CNN model not loaded'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)
        
        # Process video
        enhanced_inference = EnhancedInference(cnn_model)
        result = enhanced_inference.process_video(input_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': {
                    'original_path': input_path,
                    'enhanced_path': result['enhanced_path'],
                    'filename': filename
                },
                'metrics': result['metrics']
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/run-analytics', methods=['POST'])
def run_analytics_api():
    """Run comprehensive analytics on uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)
        
        # Run analytics
        if cnn_model is not None:
            enhanced_inference = EnhancedInference(cnn_model)
            enhanced_result = enhanced_inference.process_single_image(input_path)
            
            if enhanced_result['success']:
                # Run analytics on original and enhanced images
                analytics_result = run_analytics(
                    'single',
                    input_path,
                    enhanced_result['enhanced_path'],
                    output_dir=OUTPUT_FOLDER
                )
                
                return jsonify({
                    'success': True,
                    'data': {
                        'analytics_path': analytics_result.get('output_dir'),
                        'original_path': input_path,
                        'enhanced_path': enhanced_result['enhanced_path']
                    },
                    'metrics': enhanced_result['metrics']
                })
            else:
                return jsonify({'success': False, 'error': enhanced_result['error']}), 500
        else:
            return jsonify({'success': False, 'error': 'CNN model not loaded'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export-onnx', methods=['POST'])
def export_onnx():
    """Export model to ONNX format"""
    try:
        if cnn_model is None:
            return jsonify({'success': False, 'error': 'CNN model not loaded'}), 500
        
        data = request.get_json() or {}
        format_type = data.get('format', 'standard')
        optimization = data.get('optimization', True)
        
        # Export to ONNX
        exporter = ONNXExporter(cnn_model)
        result = exporter.export_onnx(
            output_path='onnx_models/mareye_api.onnx',
            format_type=format_type,
            optimization=optimization
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': {
                    'onnx_path': result['onnx_path'],
                    'model_size': result['model_size'],
                    'format': format_type
                }
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/deploy-jetson', methods=['POST'])
def deploy_jetson():
    """Deploy model to Jetson devices"""
    try:
        if cnn_model is None:
            return jsonify({'success': False, 'error': 'CNN model not loaded'}), 500
        
        data = request.get_json() or {}
        device = data.get('device', 'jetson_orin')
        optimization = data.get('optimization', 'tensorrt_fp16')
        
        # Deploy to Jetson
        deployment = EdgeDeployment()
        result = deployment.deploy_to_jetson(
            model=cnn_model,
            device=device,
            optimization=optimization
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': {
                    'device': device,
                    'optimization': optimization,
                    'deployment_path': result.get('deployment_path'),
                    'performance': result.get('performance')
                }
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<path:filename>')
def download_file(filename):
    """Download processed files"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if cnn_model is None:
            return jsonify({'success': False, 'error': 'CNN model not loaded'}), 500
        
        # Count model parameters
        total_params = sum(p.numel() for p in cnn_model.parameters())
        trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
        
        return jsonify({
            'success': True,
            'data': {
                'model_type': 'U-Net',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
                'architecture': 'Truncated U-Net for underwater image enhancement'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting MAR EYE CNN API Server on port {port}")
    print(f"Model loaded: {cnn_model is not None}")
    app.run(host='0.0.0.0', port=port, debug=False)
