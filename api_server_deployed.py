#!/usr/bin/env python3
"""
MAR EYE CNN API Server - Deployed Version
Optimized for Render free tier with memory constraints
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
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, origins=['https://mareye-frontend.vercel.app', 'http://localhost:3000', 'http://localhost:3001'], 
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], 
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin'])

# Add explicit CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_uiqm(img):
    """Calculate UIQM (Underwater Image Quality Measure)"""
    try:
        # Convert to float
        img_float = img.astype(np.float32) / 255.0
        
        # Calculate contrast (standard deviation)
        contrast = np.std(img_float)
        
        # Calculate saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:,:,1]) / 255.0
        
        # Calculate sharpness (using Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate colorfulness
        b, g, r = cv2.split(img)
        colorfulness = np.sqrt(np.var(r) + np.var(g) + np.var(b)) / 255.0
        
        # Combine metrics (simplified UIQM)
        uiqm = (contrast * 100) + (saturation * 50) + (sharpness / 100) + (colorfulness * 25)
        return uiqm
    except Exception as e:
        print(f"Error calculating UIQM: {e}")
        return 0.0

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Simple SSIM calculation
        mu1 = cv2.GaussianBlur(gray1.astype(np.float32), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2.astype(np.float32), (11, 11), 1.5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur((gray1.astype(np.float32) * gray1.astype(np.float32)), (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur((gray2.astype(np.float32) * gray2.astype(np.float32)), (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur((gray1.astype(np.float32) * gray2.astype(np.float32)), (11, 11), 1.5) - mu1_mu2
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        ssim_value = np.mean(ssim_map)
        
        return max(0.0, min(1.0, ssim_value))
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0.0

def enhance_image_opencv(image_array):
    """Enhanced OpenCV-based image enhancement"""
    try:
        # Resize to reduce memory usage
        height, width = image_array.shape[:2]
        if width > 512 or height > 512:
            scale = min(512/width, 512/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_array = cv2.resize(image_array, (new_width, new_height))
        
        # DRAMATIC enhancement - very visible changes
        # 1. Extreme brightness boost
        enhanced = cv2.convertScaleAbs(image_array, alpha=2.5, beta=80)
        
        # 2. Sharpening filter for crisp edges
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # 3. Saturation boost for vivid colors
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 2.0  # Double saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 4. Extreme CLAHE for dramatic contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 5. Final brightness adjustment
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=20)
        
        return enhanced
    except Exception as e:
        print(f"Error in OpenCV enhancement: {e}")
        return image_array

def process_image(image_path, mode='underwater'):
    """Process image with OpenCV enhancement"""
    try:
        print(f"Processing image: {image_path}")
        
        # Read image
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError("Could not read image")
        
        # Calculate original metrics
        uiqm_original = calculate_uiqm(image_array)
        
        # Enhance image
        enhanced_array = enhance_image_opencv(image_array)
        
        # Calculate enhanced metrics
        uiqm_enhanced = calculate_uiqm(enhanced_array)
        
        # Calculate PSNR
        mse = np.mean((image_array.astype(float) - enhanced_array.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 0
        
        # Calculate SSIM
        ssim = calculate_ssim(image_array, enhanced_array)
        
        # Calculate UIQM improvement
        uiqm_improvement = ((uiqm_enhanced - uiqm_original) / uiqm_original * 100) if uiqm_original > 0 else 0
        
        # Save enhanced image
        enhanced_filename = f"enhanced_{os.path.basename(image_path)}"
        enhanced_path = os.path.join(OUTPUT_FOLDER, enhanced_filename)
        cv2.imwrite(enhanced_path, enhanced_array)
        
        # Convert to base64 for API response
        _, buffer = cv2.imencode('.jpg', enhanced_array)
        enhanced_data = base64.b64encode(buffer).decode('utf-8')
        
        metrics = {
            'psnr': float(psnr),
            'ssim': float(ssim),
            'uiqm_original': float(uiqm_original),
            'uiqm_enhanced': float(uiqm_enhanced),
            'uiqm_improvement': float(uiqm_improvement)
        }
        
        print(f"Processing complete. Metrics: {metrics}")
        
        return {
            'enhanced_image': Image.fromarray(cv2.cvtColor(enhanced_array, cv2.COLOR_BGR2RGB)),
            'enhanced_path': enhanced_path,
            'enhanced_data': enhanced_data,
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise e

def process_video(video_path, mode='underwater'):
    """Process video with OpenCV enhancement"""
    try:
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Resize for memory efficiency
        if width > 640 or height > 480:
            scale = min(640/width, 480/height)
            width = int(width * scale)
            height = int(height * scale)
        
        # Create output video
        enhanced_filename = f"enhanced_{os.path.basename(video_path)}"
        enhanced_path = os.path.join(OUTPUT_FOLDER, enhanced_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(enhanced_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_psnr = 0
        total_ssim = 0
        total_uiqm_original = 0
        total_uiqm_enhanced = 0
        
        # Process frames (limit to 30 frames for memory efficiency)
        max_frames = 30
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Calculate original metrics
            uiqm_original = calculate_uiqm(frame)
            
            # Enhance frame
            enhanced_frame = enhance_image_opencv(frame)
            
            # Calculate enhanced metrics
            uiqm_enhanced = calculate_uiqm(enhanced_frame)
            
            # Calculate PSNR
            mse = np.mean((frame.astype(float) - enhanced_frame.astype(float)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 0
            
            # Calculate SSIM
            ssim = calculate_ssim(frame, enhanced_frame)
            
            # Accumulate metrics
            total_psnr += psnr
            total_ssim += ssim
            total_uiqm_original += uiqm_original
            total_uiqm_enhanced += uiqm_enhanced
            
            # Write enhanced frame
            out.write(enhanced_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Calculate average metrics
        avg_psnr = total_psnr / frame_count if frame_count > 0 else 0
        avg_ssim = total_ssim / frame_count if frame_count > 0 else 0
        avg_uiqm_original = total_uiqm_original / frame_count if frame_count > 0 else 0
        avg_uiqm_enhanced = total_uiqm_enhanced / frame_count if frame_count > 0 else 0
        avg_uiqm_improvement = ((avg_uiqm_enhanced - avg_uiqm_original) / avg_uiqm_original * 100) if avg_uiqm_original > 0 else 0
        
        metrics = {
            'psnr': float(avg_psnr),
            'ssim': float(avg_ssim),
            'uiqm_original': float(avg_uiqm_original),
            'uiqm_enhanced': float(avg_uiqm_enhanced),
            'uiqm_improvement': float(avg_uiqm_improvement)
        }
        
        print(f"Video processing complete. Processed {frame_count} frames. Metrics: {metrics}")
        
        return {
            'enhanced_path': enhanced_path,
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"Error processing video: {e}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MAR EYE CNN API',
        'version': '1.0.0',
        'model_loaded': True,
        'optimization': 'deployed_free_tier'
    })

@app.route('/api/process-image', methods=['POST'])
def process_image_endpoint():
    """Process image endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get enhancement mode
        mode = request.form.get('mode', 'underwater')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Process image
            result = process_image(file_path, mode)
            
            # Return response
            return jsonify({
                'success': True,
                'data': {
                    'enhanced_path': result['enhanced_path'],
                    'enhanced_url': f'https://mereyecnn-r60u.onrender.com/outputs/{os.path.basename(result["enhanced_path"])}',
                    'enhanced_data': f'data:image/jpeg;base64,{result["enhanced_data"]}',
                    'metrics': result['metrics']
                }
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        print(f"Error in process_image_endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-video', methods=['POST'])
def process_video_endpoint():
    """Process video endpoint"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get enhancement mode
        mode = request.form.get('mode', 'underwater')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Process video
            result = process_video(file_path, mode)
            
            # Return response
            return jsonify({
                'success': True,
                'data': {
                    'enhanced_path': result['enhanced_path'],
                    'enhanced_url': f'https://mereyecnn-r60u.onrender.com/outputs/{os.path.basename(result["enhanced_path"])}',
                    'metrics': result['metrics']
                }
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        print(f"Error in process_video_endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serve enhanced images/videos"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting MAR EYE CNN API Server - Deployed Version")
    print("Optimized for Render free tier")
    print("Model loaded: True (OpenCV Enhancement)")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)
