"""
Medical Diagnostic Platform - Tumor Classification System
---------------------------------------------------------
A Flask-based web service for breast tumor classification using
machine learning. Implements the WDBC (Wisconsin Diagnostic Breast Cancer)
feature set for binary classification.

Core Capabilities:
- Member authentication and session handling
- Feature-based tumor analysis
- Diagnostic history management
- Persistent data storage via SQLite
"""

import os
import json
import sqlite3
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow import keras


# ------------------------------------
# Flask Application Setup
# ------------------------------------

webapp = Flask(__name__)
webapp.config['SECRET_KEY'] = 'secure-token-replace-in-production-env'
webapp.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Asset locations
CLASSIFIER_PATH = 'Model.h5'
NORMALIZER_PATH = 'scaler.pkl'
ATTRIBUTE_LIST_PATH = 'feature_names.json'


# ------------------------------------
# Data Persistence Layer
# ------------------------------------

def setup_database():
    """
    Establishes the SQLite schema for members and diagnostic records.
    Creates tables if they don't already exist.
    """
    connection = sqlite3.connect('database.db')
    db_cursor = connection.cursor()
    
    # Members storage
    db_cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Diagnostic records storage
    db_cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_result TEXT NOT NULL,
            confidence REAL NOT NULL,
            features TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    connection.commit()
    connection.close()
    print("[OK] Database schema initialized")


def establish_db_connection():
    """Returns a configured database connection with row factory."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------
# ML Components Loader
# ------------------------------------

def initialize_ml_components():
    """
    Loads the trained neural network, data normalizer, and feature definitions.
    Returns tuple of (classifier, normalizer, attributes).
    """
    classifier = None
    normalizer = None
    attributes = []
    
    # Load neural network
    try:
        classifier = keras.models.load_model(CLASSIFIER_PATH)
        print("[OK] Neural network classifier loaded")
    except Exception as err:
        print(f"[FAIL] Classifier loading error: {err}")
    
    # Load data normalizer
    try:
        normalizer = joblib.load(NORMALIZER_PATH)
        print("[OK] Data normalizer loaded")
    except Exception as err:
        print(f"[FAIL] Normalizer loading error: {err}")
    
    # Load attribute definitions
    try:
        with open(ATTRIBUTE_LIST_PATH, 'r') as attr_file:
            attributes = json.load(attr_file)
        print(f"[OK] Loaded {len(attributes)} feature attributes")
    except Exception as err:
        print(f"[FAIL] Attribute list loading error: {err}")
    
    return classifier, normalizer, attributes


# Global ML components
classifier = None
normalizer = None
attributes = []


# ------------------------------------
# Authentication Endpoints
# ------------------------------------

@webapp.route('/')
def home_page():
    """Renders the main interface based on authentication status."""
    if 'user_id' in session:
        return render_template('index.html', 
                             logged_in=True, 
                             username=session.get('username'),
                             feature_names=attributes)
    return render_template('index.html', logged_in=False)


@webapp.route('/register', methods=['POST'])
def create_account():
    """Handles new member registration with validation."""
    try:
        payload = request.get_json()
        member_name = payload.get('username')
        member_email = payload.get('email')
        member_pass = payload.get('password')
        
        # Input validation
        if not member_name or not member_email or not member_pass:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        # Secure password storage
        hashed_pass = generate_password_hash(member_pass)
        
        # Persist to database
        connection = establish_db_connection()
        try:
            connection.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (member_name, member_email, hashed_pass)
            )
            connection.commit()
            
            # Retrieve new member ID
            member_record = connection.execute(
                'SELECT id FROM users WHERE username = ?', (member_name,)
            ).fetchone()
            
            # Initialize session
            session['user_id'] = member_record['id']
            session['username'] = member_name
            
            return jsonify({'success': True, 'message': 'Registration successful'})
        
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'message': 'Username or email already exists'}), 400
        
        finally:
            connection.close()
    
    except Exception as err:
        return jsonify({'success': False, 'message': str(err)}), 500


@webapp.route('/login', methods=['POST'])
def authenticate_member():
    """Validates credentials and establishes session."""
    try:
        payload = request.get_json()
        member_name = payload.get('username')
        member_pass = payload.get('password')
        
        # Input validation
        if not member_name or not member_pass:
            return jsonify({'success': False, 'message': 'Username and password are required'}), 400
        
        # Credential verification
        connection = establish_db_connection()
        member_record = connection.execute(
            'SELECT id, username, password_hash FROM users WHERE username = ?',
            (member_name,)
        ).fetchone()
        connection.close()
        
        if member_record and check_password_hash(member_record['password_hash'], member_pass):
            # Establish session
            session['user_id'] = member_record['id']
            session['username'] = member_record['username']
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
    
    except Exception as err:
        return jsonify({'success': False, 'message': str(err)}), 500


@webapp.route('/logout')
def terminate_session():
    """Clears session and redirects to landing page."""
    session.clear()
    return redirect(url_for('home_page'))


# ------------------------------------
# Diagnostic Analysis Endpoints
# ------------------------------------

@webapp.route('/predict', methods=['POST'])
def run_diagnostic():
    """Processes feature input and returns classification result."""
    # Authentication check
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    # ML components availability check
    if classifier is None or normalizer is None:
        return jsonify({'success': False, 'message': 'Model not loaded'}), 500
    
    try:
        # Extract feature values from request
        payload = request.get_json()
        feature_data = payload.get('features', {})
        
        if not feature_data:
            return jsonify({'success': False, 'message': 'No features provided'}), 400
        
        # Construct ordered feature vector
        feature_vector = []
        for attr_name in attributes:
            attr_value = feature_data.get(attr_name)
            if attr_value is None or attr_value == '':
                return jsonify({'success': False, 'message': f'Missing value for {attr_name}'}), 400
            try:
                feature_vector.append(float(attr_value))
            except ValueError:
                return jsonify({'success': False, 'message': f'Invalid value for {attr_name}'}), 400
        
        # Prepare input tensor
        input_tensor = np.array(feature_vector).reshape(1, -1)
        
        # Normalize features
        normalized_input = normalizer.transform(input_tensor)
        
        # Execute classification
        raw_probability = classifier.predict(normalized_input, verbose=0)[0][0]
        
        # Interpret classification output
        if raw_probability > 0.5:
            diagnosis = 'malignant'
            diagnosis_label = 'Malignant (Cancerous)'
            certainty = float(raw_probability * 100)
        else:
            diagnosis = 'benign'
            diagnosis_label = 'Benign (Non-cancerous)'
            certainty = float((1 - raw_probability) * 100)
        
        # Persist diagnostic record
        connection = establish_db_connection()
        connection.execute(
            'INSERT INTO predictions (user_id, prediction_result, confidence, features) VALUES (?, ?, ?, ?)',
            (session['user_id'], diagnosis, certainty, json.dumps(feature_data))
        )
        connection.commit()
        connection.close()
        
        # Return classification result
        return jsonify({
            'success': True,
            'result': diagnosis_label,
            'confidence': round(certainty, 2),
            'raw_probability': float(raw_probability)
        })
    
    except Exception as err:
        return jsonify({'success': False, 'message': f'Error processing prediction: {str(err)}'}), 500


@webapp.route('/history')
def fetch_diagnostic_history():
    """Retrieves past diagnostic records for current member."""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    try:
        connection = establish_db_connection()
        records = connection.execute(
            'SELECT prediction_result, confidence, created_at FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 20',
            (session['user_id'],)
        ).fetchall()
        connection.close()
        
        # Transform to list format
        history_records = []
        for record in records:
            history_records.append({
                'result': record['prediction_result'],
                'confidence': round(record['confidence'], 2),
                'date': record['created_at']
            })
        
        return jsonify({'success': True, 'history': history_records})
    
    except Exception as err:
        return jsonify({'success': False, 'message': str(err)}), 500


@webapp.route('/get_feature_names')
def list_feature_attributes():
    """Returns the complete list of expected feature attributes."""
    return jsonify({'success': True, 'features': attributes})


# ------------------------------------
# Error Management
# ------------------------------------

@webapp.errorhandler(404)
def page_not_found(error):
    """Custom 404 handler."""
    return render_template('index.html'), 404


@webapp.errorhandler(500)
def internal_error(error):
    """Custom 500 handler."""
    return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ------------------------------------
# Application Bootstrap
# ------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("TUMOR CLASSIFICATION DIAGNOSTIC PLATFORM")
    print("=" * 70)
    
    # Database initialization
    print("\n[Step 1/2] Setting up database...")
    setup_database()
    
    # ML components initialization
    print("\n[Step 2/2] Loading ML components...")
    classifier, normalizer, attributes = initialize_ml_components()
    
    if classifier is None or normalizer is None or not attributes:
        print("\n[WARNING] Some components failed to load!")
        print("Required files:")
        print("  - Model.h5")
        print("  - scaler.pkl")
        print("  - feature_names.json")
        print("\nApplication will start but predictions will not function.\n")
    
    print("\n" + "=" * 70)
    print("[READY] Application initialized successfully")
    print("=" * 70)
    print("\nServer starting...")
    print("   URL: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop\n")
    
    # Launch server
    webapp.run(debug=True, host='0.0.0.0', port=5000)