#!/usr/bin/env python3
"""
Simple test Flask server
"""

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:3001'])

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Test Flask Server',
        'version': '1.0.0'
    })

@app.route('/test-cors', methods=['GET'])
def test_cors():
    """Test CORS endpoint"""
    return jsonify({
        'message': 'CORS is working!',
        'status': 'success'
    })

if __name__ == '__main__':
    print("Starting test Flask server on port 8000...")
    app.run(host='127.0.0.1', port=8000, debug=True)












