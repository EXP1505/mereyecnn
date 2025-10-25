#!/usr/bin/env python3
"""
Ultra-minimal startup script for Render deployment
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the ultra-minimal server
from api_server_ultra_minimal import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting ultra-minimal MAR EYE API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
