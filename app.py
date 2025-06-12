from flask import Flask, render_template, request, send_file, jsonify, session, redirect, url_for, flash
import cv2
import numpy as np
import os
from PIL import Image
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from werkzeug.utils import secure_filename
from datetime import datetime
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# MongoDB Connection
client = MongoClient(os.getenv('MONGODB_URI'))
db = client['image_colorization']
users = db['users']

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# filename paths
PROTOTXT_PATH = "models/colorize.prototext"
MODEL_PATH = "models/release.caffemodel"
POINTS_PATH = "models/pts_in_hull.npy"

# Load model
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
pts = np.load(POINTS_PATH)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if users.find_one({'email': email}):
            flash('Email already exists', 'error')
            return redirect(url_for('signup'))
        
        user = {
            'name': name,
            'email': email,
            'password': password,  # In production, use proper password hashing
            'created_at': datetime.utcnow()
        }
        
        users.insert_one(user)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = users.find_one({'email': email})
        if user and user['password'] == password:  # In production, use proper password hashing
            session['user_id'] = str(user['_id'])
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    if 'user_id' not in session:
        flash('Please login to access this page', 'error')
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401

    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    image = cv2.imread(filepath)
    if image is None:
        return "Invalid image format", 400

    original_size = (image.shape[1], image.shape[0])
    lab = cv2.cvtColor(image.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
    L_original = lab[:, :, 0]  # Extract L channel (original size)
    L_input = cv2.resize(L_original, (224, 224)) - 50  # Resize for model
    net.setInput(cv2.dnn.blobFromImage(L_input))
    ab_base = net.forward()[0].transpose((1, 2, 0))
    ab_base = cv2.resize(ab_base, original_size)

    result_paths = []
    for i in range(8):
        ab = ab_base * (1 + (i - 4) * 0.1)
        lab_output = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)
        colorized = (np.clip(colorized, 0, 1) * 255).astype("uint8")
        result_path = os.path.join(RESULT_FOLDER, f"colorized_{i}_{filename}")
        cv2.imwrite(result_path, colorized)
        result_paths.append(f"/static/results/colorized_{i}_{filename}")

    return jsonify({"images": result_paths})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


