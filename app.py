from flask import Flask, request, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import re
from flask_cors import CORS
import os
import traceback

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///auth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize extensions
db = SQLAlchemy(app)

# Configure CORS - Allow all origins during development
CORS(app)  # Enable CORS for all routes

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5000'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'fullname': self.fullname,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/profile')
def serve_profile():
    return send_from_directory('static', 'profile.html')

@app.route('/onboarding')
def serve_onboarding():
    return send_from_directory('static', 'onboarding.html')

@app.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if request.is_json:
            data = request.get_json()
        else:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['fullname', 'email', 'password']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        if not is_valid_email(data['email']):
            return jsonify({'error': 'Invalid email format'}), 400

        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 400

        if len(data['password']) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        user = User(
            fullname=data['fullname'],
            email=data['email'],
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
            is_active=True
        )
        user.set_password(data['password'])
        
        try:
            db.session.add(user)
            db.session.commit()
        except Exception as db_error:
            db.session.rollback()
            return jsonify({'error': 'Database error during registration'}), 500
        
        try:
            session['user_id'] = user.id
        except Exception as session_error:
            # Continue even if session creation fails
            pass
        
        return jsonify({
            'message': 'Registration successful',
            'redirect': '/onboarding',
            'user': user.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed: ' + str(e)}), 500

@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if request.is_json:
            data = request.get_json()
        else:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if not all(key in data for key in ['email', 'password']):
            return jsonify({'error': 'Missing email or password'}), 400

        user = User.query.filter_by(email=data['email']).first()
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        if not user.check_password(data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401

        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        try:
            user.last_login = datetime.utcnow()
            db.session.commit()
        except Exception as db_error:
            db.session.rollback()
            return jsonify({'error': 'Database error during login'}), 500
        
        try:
            session['user_id'] = user.id
        except Exception as session_error:
            # Continue even if session creation fails
            pass
        
        return jsonify({
            'message': 'Login successful',
            'redirect': '/profile',
            'user': user.to_dict()
        }), 200

    except Exception as e:
        return jsonify({'error': 'Login failed: ' + str(e)}), 500

@app.route('/api/logout')
def logout():
    response_data = {'error': None, 'message': None}
    try:
        session.pop('user_id', None)
        response_data['message'] = 'Logged out successfully'
        return jsonify(response_data), 200
    except Exception as e:
        print(f"Logout error: {str(e)}")
        response_data['error'] = 'Logout failed'
        return jsonify(response_data), 500

@app.route('/api/check-auth')
def check_auth():
    response_data = {'error': None, 'message': None, 'user': None}
    try:
        if 'user_id' not in session:
            response_data['error'] = 'Not authenticated'
            return jsonify(response_data), 401
        
        user = User.query.get(session['user_id'])
        if not user:
            response_data['error'] = 'User not found'
            return jsonify(response_data), 404
            
        response_data['message'] = 'Authenticated'
        response_data['user'] = user.to_dict()
        return jsonify(response_data), 200
    except Exception as e:
        print(f"Auth check error: {str(e)}")
        response_data['error'] = 'Authentication check failed'
        return jsonify(response_data), 500

# Create database tables
with app.app_context():
    try:
        db.create_all()
        print("Database tables created successfully")
    except Exception as e:
        print("Error creating database tables:", str(e))

if __name__ == '__main__':
    app.run(debug=True)
