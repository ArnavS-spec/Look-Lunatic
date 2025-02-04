import os
require('dotenv').config()
from dotenv import load_dotenv
load_dotenv()
from waitress import serve
from app import app

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)
