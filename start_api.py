#!/usr/bin/env python3
"""
Start script for MAR EYE CNN API Server
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the API server
from api_server import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting MAR EYE CNN API Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
