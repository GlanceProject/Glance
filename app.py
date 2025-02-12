# ==== Import Section ====
import cv2  # OpenCV library for face detection and image processing
import numpy as np  # Numerical operations and array handling
from PIL import Image  # Image manipulation (used in QR code generation)
from flask import (  Flask, render_template,
                   Response, request, send_file, # Flask components for web application
                   redirect, url_for, flash,
                   session, jsonify)
from flask_socketio import SocketIO, emit
from werkzeug.exceptions import BadRequest, Unauthorized # Flask exceptions
import io  # Input/output operations (used for QR code in-memory storage)
import sqlite3  # SQLite database operations
from typing import List, Optional, Tuple, Generator # Type hinting support
from pathlib import Path  # Object-oriented file system paths
import time  # Time-related functions
from time import sleep  # Time-related functions
from datetime import datetime, timedelta  # Time delta handling
import socket  # Network connections (for IP detection)
import threading  # Concurrent execution management
import qrcode  # QR code creation library
import secrets


# ==== Flask Application Setup ====
app = Flask(__name__)  # Create Flask application instance
app.secret_key = 'complex_key_here_$%^&*()_12345'  # Encryption key for session security
socketio = SocketIO(app)

active_mobile_count = 0
# Host connection handler
@socketio.on('connect', namespace='/host')
def handle_host_connect():
    print('Host connected')

@socketio.on('disconnect', namespace='/host')
def handle_disconnect():
    print('Host disconnected')

# Updated handle_navigation function
@socketio.on('navigation_event', namespace='/host')
def handle_navigation(data):
    if data.get('isMobile'):
        socketio.emit(
            'navigation_sync',
            {'path': data['path']},
            include_self=False,
            namespace='/host'
        )
        print(f"Mobile navigation sync: {data['path']}")

# Updated handle_page_load function
@socketio.on('page_loaded', namespace='/host')
def handle_page_load(data):
    if data.get('isMobile'):
        socketio.emit(
            'navigation_sync',
            {'path': data['path']},
            include_self=False,
            namespace='/host'
        )
        print(f"Mobile page load sync: {data['path']}")

@socketio.on('registration_complete', namespace='/host')
def handle_registration_complete(data):
    # Broadcast the navigation sync event to all connected mobile clients
    socketio.emit(
        'navigation_sync',
        {'path': '/login'},
        namespace='/host'
    )
    print("Registration complete. Broadcasting navigation to /login")
    
# ==== Camera Access Control ====
camera_lock = threading.Lock()  # Prevent concurrent camera access

# ==== Database Management ====
def get_db():
    """Create new database connection"""
    return sqlite3.connect('users.db')

def init_db():
    """Initialize database schema"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  pin TEXT)''')
    
    # Schedules table
    c.execute('''CREATE TABLE IF NOT EXISTS schedules 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  title TEXT NOT NULL,
                  note TEXT,
                  schedule_date DATE ,
                  schedule_time TIME ,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    # Registrations table 
    c.execute('''CREATE TABLE IF NOT EXISTS registrations     
                (token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                pin TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                captime REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
   
    conn.commit()
    conn.close()

# Always initialize tables on app start
try:
    init_db()
    print("Database schema verified/updated")
except Exception as e:
    print(f"Database initialization error: {e}")

# Load Haar cascades for facial feature detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_recognizer():
    """Load pre-trained recognition model"""
    model_path = Path('models/trained_model.yml')
    if model_path.exists():
        recognizer.read(str(model_path))  # Load existing model
    else:
        print("No trained model found.")  # Handle missing model

# ==== Image Processing Pipeline ====
def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (256,256)) -> np.ndarray:
    """Standardize images for training with:
    1. Grayscale conversion
    2. Resizing
    3. Histogram equalization
    4. Noise reduction
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to target dimensions
    resized = cv2.resize(image, target_size)
    
    # Enhance contrast
    equalized = cv2.equalizeHist(resized)
    
    # Reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    return blurred

def align_face(image: np.ndarray) -> Optional[np.ndarray]:
    try:
        # Parameters for eye detection
        eye_cascade_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 3,
            'minSize': (20, 20),  # Minimum reasonable eye size
        }

        # Detect faces in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            return None

        # Select the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        face_gray = gray[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_gray, **eye_cascade_params)
        if len(eyes) < 2:
            return None

        # Sort eyes by x-coordinate (left to right)
        eyes = sorted(eyes[:2], key=lambda e: e[0])
        eye1, eye2 = eyes

        # Calculate eye centers
        eye1_center = (x + eye1[0] + eye1[2] // 2, y + eye1[1] + eye1[3] // 2)
        eye2_center = (x + eye2[0] + eye2[2] // 2, y + eye2[1] + eye2[3] // 2)

        # Calculate rotation angle and midpoint
        dx = eye2_center[0] - eye1_center[0]
        dy = eye2_center[1] - eye1_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        # Ensuring the midpoint values are integers
        midpoint = (
            int((eye1_center[0] + eye2_center[0]) // 2),
            int((eye1_center[1] + eye2_center[1]) // 2)
        )

        # Rotate the image to align the eyes
        M = cv2.getRotationMatrix2D(midpoint, angle, scale=1)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Crop the aligned face
        face_aligned = aligned[y:y + h, x:x + w]

        # Quality checks
        if face_aligned.mean() < 30:  # Reject overly dark images
            return None

        # Blur detection
        laplacian_var = cv2.Laplacian(cv2.cvtColor(face_aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if laplacian_var < 100:  # Reject blurry images
            return None

        # Check for eye presence post-alignment
        aligned_gray = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2GRAY)
        aligned_eyes = eye_cascade.detectMultiScale(aligned_gray, **eye_cascade_params)
        if len(aligned_eyes) < 2:
            return None

        # Check aspect ratio
        h, w = face_aligned.shape[:2]
        if not (0.7 <= w / h <= 1.3):  # Valid face aspect ratio
            return None

        # Resize to standard dimensions
        return cv2.resize(face_aligned, (256, 256))

    except Exception as e:
        print(f"Error aligning face: {e}")
        return None
    

def recognize_face() -> Optional[str]:
    """Recognize face and return username if confidence is high enough"""
    try:
        with camera_lock:
            cap = cv2.VideoCapture(2)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return None

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                return None

            aligned_face = align_face(frame)
            if aligned_face is None:
                print("Error: Could not align face")
                return None

            preprocessed = preprocess_image(aligned_face)
            
            # Create fresh recognizer instance
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            model_path = 'models/trained_model.yml'

            if not Path(model_path).exists():
                print("Error: No trained model found")
                return None

            try:
                recognizer.read(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                return None

            try:
                user_id, confidence = recognizer.predict(preprocessed)
                print(f"Predicted user ID: {user_id} with confidence {confidence}")

                if confidence < 35:  # Adjust this threshold as needed
                    with get_db() as conn:
                        c = conn.cursor()
                        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                        result = c.fetchone()
                        if result:
                            return result[0]
            except Exception as e:
                print(f"Prediction error: {e}")
                return None
            finally:
                cap.release()

        return None
        
    except Exception as e:
        print(f"Error in recognize_face: {e}")
        return None


@app.route('/video_feed')
def video_feed():
    """Robust video feed with improved error handling"""
    # Initialize recognizer outside the generator
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read('models/trained_model.yml')
    except :
        print("Model load error")

    def generate_frames():
        with camera_lock:
            cap = cv2.VideoCapture(2)
            if not cap.isOpened():
                print("Cannot open camera")
                return

        try:
            frame_counter = 0
            while True:
                success, frame = cap.read()
                if not success:
                    print("Failed to grab frame")
                    break

                frame_counter += 1
                
                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(60, 60)
                )

                for (x, y, w, h) in faces:
                    try:
                        # Face processing
                        face_roi = frame[y:y+h, x:x+w]
                        aligned_face = align_face(face_roi)
                        
                        if aligned_face is None:
                            aligned_face = cv2.resize(face_roi, (256, 256))
                            
                        preprocessed = preprocess_image(aligned_face)
                        
                        # Prediction
                        user_id, confidence = recognizer.predict(preprocessed)
                        username = "Unknown"
                        if confidence < 35 :
                            confidence = int(confidence)
                            with get_db() as conn:
                                c = conn.cursor()
                                c.execute("SELECT username FROM users WHERE id=?", (user_id,))
                                result = c.fetchone()
                                username = result[0] if result else "Unknown"

                            # Create label based on recognition status
                            if username == "Unknown":
                                label = "Unknown"
                            else:
                                label = f"{username} ({confidence})"

                            # Draw annotations
                            color = (0, 255, 0) if username != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        else:
                            label = "Unknown"
                            # Draw annotations
                            color = (0, 255, 0) if username != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    except Exception as e:
                        pass

                # Single yield after processing
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception:
            pass
        finally:
            cap.release()

    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==== Web Routes ====
@app.route('/')
def index():
    """Main entry point with device detection."""
    user_agent = request.user_agent.string
    is_mobile = 'Mobi' in user_agent
    print(f"Current session username: {session.get('username')}")  # Debug log
    return render_template('index.html', is_mobile=is_mobile)

@app.route('/api/recognize_face', methods=['GET'])
def api_recognize_face():
    """API endpoint to recognize a face."""
    recognized_user = recognize_face()
    if recognized_user:
        session.clear()  # Clear stale session data
        session['username'] = recognized_user
        print(f"Recognized user: {recognized_user}")  # Debug log
        socketio.emit('user_changed', {'username': recognized_user}, namespace='/host')
        return redirect(url_for('dashboard_user'))
    return jsonify({'status': 'no_user_recognized'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route with dual authentication (face recognition + PIN)."""
    is_mobile = 'Mobi' in request.user_agent.string
    recognized_user = recognize_face()
    if recognized_user:
        session['username'] = recognized_user
        return redirect(url_for('dashboard_user'))
    if request.method == 'POST':
        username = request.form['username']
        pin = request.form['pin']
        with get_db() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND pin = ?", (username, pin))
            user = c.fetchone()
            if user:
                session['username'] = username
                return redirect(url_for('dashboard_user'))
            else:
                flash("Invalid username or PIN.", "error")
    return render_template('login.html', is_mobile=is_mobile)

# ==== Registration Routes ====

@app.route('/register', methods=['GET', 'POST'])
def register():
         
    is_mobile = 'Mobi' in request.user_agent.string
    if is_mobile and request.method == 'GET':
        socketio.emit('mobile_entered_register', namespace='/host')
    if request.method == 'POST':
        username = request.form.get('username')
        pin = request.form.get('pin')
        
        if not username or not pin:
            flash("All fields are required", "error")
            return redirect(url_for('register'))

        token = secrets.token_urlsafe(32)
        try:
            with get_db() as conn:
                # Use context manager for transaction
                with conn:
                    # In register route when creating new registration
                    conn.execute('''INSERT INTO registrations 
                                (token, username, pin, captime, progress) 
                                VALUES (?, ?, ?, ?, ?)''',
                            (token, username, pin, time.time(), 0))  # Initialize progress at 0
                    socketio.emit('new_registration',  # Key WebSocket trigger
                            {'token': token, 'username': username},
                            namespace='/host')        
                print(f"[DEBUG] Created registration token: {token}")  # Add debug log
                return redirect(url_for('registration_capture', token=token))

        except sqlite3.IntegrityError as e:
            print(f"[ERROR] Database error: {str(e)}")  # Detailed error logging
            flash("Username already exists or registration in progress", "error")
            return redirect(url_for('register'))

        except Exception as e:
            print(f"[ERROR] Registration failed: {str(e)}")
            flash("Registration failed. Please try again.", "error")
            return redirect(url_for('register'))

    return render_template('register.html' , is_mobile=is_mobile)

@app.route('/check_active_registration')
#Define a function to check if there is an active registration
def check_active_registration():
    #Connect to the database
    with get_db() as conn:
        #Execute a query to select the token from the registrations table where the progress is less than 30 and the created_at date is within the last 15 minutes
        reg = conn.execute('''SELECT token FROM registrations 
                            WHERE progress < 30 
                            AND datetime(created_at) > datetime('now', '-15 minutes')
                            LIMIT 1''').fetchone()
    #Return a jsonified response with the active status and the token
    return jsonify({'active': bool(reg), 'token': reg[0] if reg else None})

@app.route('/registration_capture/<token>')
def registration_capture(token):
    try:
        # Check if the user is on a mobile device
        is_mobile = 'Mobi' in request.user_agent.string
        with get_db() as conn:
            reg = conn.execute('''SELECT username, progress, captime
                                FROM registrations 
                                WHERE token = ? 
                                AND datetime(created_at) > datetime('now', '-15 minutes')''',
                             (token,)).fetchone()

            if not reg:
                print(f"[WARNING] Invalid token: {token}")  # Log invalid tokens
                flash("Invalid or expired registration session", "error")
                return redirect(url_for('register'))

            print(f"[DEBUG] Valid registration session: {token}")  # Success log
            return render_template('registration_capture.html',
                         token=token,
                         username=reg[0],
                         progress=reg[1],
                         captime=reg[2],
                         timestamp=datetime.now().timestamp(),
                         is_mobile=is_mobile)

    except Exception as e:
        print(f"[ERROR] Token validation failed: {str(e)}")
        flash("Registration error. Please start over.", "error")
        return redirect(url_for('register'))

@app.route('/reg_video_feed/<token>')
def reg_video_feed(token):
    def generate():
        try:
            # Fetch registration info
            with get_db() as conn:
                reg = conn.execute(
                    '''SELECT username, pin, progress FROM registrations WHERE token = ?''', 
                    (token,)
                ).fetchone()

            if not reg:
                print(f"[ERROR] Invalid token: {token}")
                return

            username, pin, progress = reg

            # Open the camera
            with camera_lock:
                cap = cv2.VideoCapture(2)
                if not cap.isOpened():
                    print("[ERROR] Could not open video source")
                    return

                # Camera optimization
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                last_capture_time = 0
                try:
                    while True:
                        success, frame = cap.read()
                        if not success:
                            break

                        ret, buffer = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                        current_time = time.time()
                        if current_time - last_capture_time >= 1.75:
                            last_capture_time = current_time

                            with get_db() as conn:
                                progress = conn.execute(
                                    '''SELECT progress FROM registrations WHERE token = ?''',
                                    (token,)
                                ).fetchone()[0]

                            if progress >= 30:
                                # Create user first
                                with get_db() as conn:
                                    try:
                                        # Start a transaction
                                        conn.execute('BEGIN')
                                        
                                        # Insert the user
                                        cursor = conn.execute(
                                            '''INSERT INTO users (username, pin) 
                                               VALUES (?, ?)''', 
                                            (username, pin)
                                        )
                                        
                                        # Get the new user's ID
                                        user_id = cursor.lastrowid
                                        
                                        if not user_id:
                                            raise Exception("Failed to create user")
                                            
                                        # Delete from registrations
                                        conn.execute(
                                            'DELETE FROM registrations WHERE token = ?', 
                                            (token,)
                                        )
                                        
                                        # Commit the transaction
                                        conn.commit()
                                        print(f"Successfully created user {username} with ID {user_id}")
                                        sleep(1)                                        
                                        # Train the model only after successful user creation
                                        train_model(username)
                                        socketio.emit('registration_complete', namespace='/host')
                                        break
                                        
                                    except sqlite3.IntegrityError as e:
                                        conn.execute('ROLLBACK')
                                        print(f"Database error during user creation: {e}")
                                        raise
                                    except Exception as e:
                                        conn.execute('ROLLBACK')
                                        print(f"Error during user creation: {e}")
                                        raise
                                break

                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                            if len(faces) > 0:
                                for (x, y, w, h) in faces:
                                    if progress >= 30:
                                        continue

                                    face_img = frame[y:y+h, x:x+w]
                                    aligned_result = align_face(face_img)
                                    aligned = aligned_result if aligned_result is not None else face_img
                                    aligned_preprocessed = preprocess_image(aligned)
                                    user_dir = Path(f"static/faces/{username}")
                                    user_dir.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(str(user_dir / f"{progress}.jpg"), aligned_preprocessed)

                                    with get_db() as conn:
                                        conn.execute(
                                            '''UPDATE registrations 
                                               SET progress = ?, captime = ? 
                                               WHERE token = ?''',
                                            (progress + 1, last_capture_time, token)
                                        )
                                        conn.commit()
                                    break

                finally:
                    cap.release()

        except Exception as e:
            print(f"[ERROR] Video feed failed: {e}")
            raise

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

#/registration_progress route
@app.route('/registration_progress/<token>')
def registration_progress(token):
    """Check registration progress and calculate next capture timing"""
    try:
        with get_db() as conn:
            # Get the latest registration data
            reg = conn.execute('''
                SELECT progress, captime, created_at 
                FROM registrations 
                WHERE token = ? 
                AND datetime(created_at) > datetime('now', '-15 minutes')
            ''', (token,)).fetchone()

            if not reg:
                return jsonify({
                    'progress': 0,
                    'total': 30,
                    'captime': 0,
                    'error': 'Registration not found or expired'
                }), 404

            progress, last_capture, created_at = reg
            current_time = time.time()

            # Calculate the waiting period based on progress
            if progress == 0:
                # Initial 5-second preparation time
                waiting_period = 5.0
            else:
                # 1.5 seconds between subsequent captures
                waiting_period = 2

            # Calculate remaining time until next capture
            if last_capture:
                time_since_capture = current_time - last_capture
                remaining_time = max(waiting_period - time_since_capture, 0)
            else:
                # If no captures yet, use time since registration created
                remaining_time = max(waiting_period - (current_time - time.mktime(datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S').timetuple())), 0)

            return jsonify({
                'progress': progress,
                'total': 30,
                'captime': remaining_time,
                'waiting_period': waiting_period
            })

    except Exception as e:
        print(f"Error in registration_progress: {str(e)}")
        return jsonify({
            'progress': 0,
            'total': 30,
            'captime': 0,
            'error': str(e)
        }), 500

def get_user_id(username: str) -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        return result[0] if result else -1
    
# ==== Train Model ====
def train_model(username: str = None) -> None:
    """Train face recognition model with all users' images"""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = Path('models/trained_model.yml')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    face_samples = []
    ids = []
    target_size = (256, 256)

    try:
        faces_dir = Path('static/faces')
        # Load existing model if available
        if model_path.exists():
            recognizer.read(str(model_path))

        # If a username is provided, only train for that user
        if username:
            user_dir = faces_dir / username
            if not user_dir.exists():
                raise ValueError(f"No images found for user: {username}")
            user_id = get_user_id(username)
            if user_id == -1:
                raise ValueError(f"User {username} not found in the database")
        else:
            user_dir = faces_dir

        # Iterate through user directories
        for user_folder in user_dir.iterdir() if username is None else [user_dir]:
            if user_folder.is_dir():
                current_username = user_folder.name
                current_user_id = get_user_id(current_username)
                if current_user_id == -1:
                    continue  # Skip if user not in DB

                # for each image in user folder
                for img_path in user_folder.glob('*.jpg'):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    face_samples.append(img)
                    ids.append(current_user_id)

        # Validate data
        if len(face_samples) == 0:
            raise ValueError("No valid training images found.")

        # Train or update the model
        if model_path.exists():
            recognizer.update(face_samples, np.array(ids))
        else:
            recognizer.train(face_samples, np.array(ids))
        
        recognizer.save(str(model_path))
        print(f"Model trained/updated with {len(face_samples)} samples")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

# ==== Dashboard Views ====
def get_user_schedules(user_id):
    """Helper function to fetch schedules for a given user ID."""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('''SELECT id, title, note, schedule_date, schedule_time 
                         FROM schedules 
                         WHERE user_id = ? 
                         ORDER BY schedule_date, schedule_time''', (user_id,))
            return [
                {'id': row[0], 'title': row[1], 'note': row[2], 
                 'date': row[3], 'time': row[4]}
                for row in c.fetchall()
            ]
    except Exception as e:
        # Log the error and return an empty list
        print(f"Error fetching schedules: {e}")
        return []
    
@app.route('/dashboard/user')
def dashboard_user():
    """Redirect users to appropriate dashboard based on device type."""
    username = session.get('username')
    print(f"Current session username: {username}")  # Debug log
    if not username:
        return redirect(url_for('login'))
    
    with get_db() as conn:
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        user_row = c.fetchone()
        if not user_row:
            return redirect(url_for('login'))
        
        user_id = user_row[0]
        schedules = get_user_schedules(user_id)
    
    is_mobile = 'Mobi' in request.user_agent.string
    template_name = 'dashboard.html' if is_mobile else 'RaspDashboard.html'
    return render_template(template_name, username=username, schedules=schedules)

@app.route('/dashboard/optimization')
def optimization_page():
    """Optimization page"""
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))
    
    try:
        with get_db() as conn:
            c = conn.cursor()
            # Fetch user ID based on username
            c.execute('SELECT id FROM users WHERE username = ?', (username,))
            user_row = c.fetchone()
            if not user_row:
                return redirect(url_for('login'))
            
            user_id = user_row[0]
            # Retrieve user's schedules using the helper function
            schedules = get_user_schedules(user_id)
        
        # Render the optimization page
        return render_template('optimization.html', 
                               username=username, 
                               schedules=schedules)
    except Exception as e:
        # Handle unexpected errors gracefully
        return jsonify({'error': str(e)}), 500
# ==== Schedule Management ====
import traceback

@app.route('/schedules', methods=['GET', 'POST', 'DELETE'])
def manage_schedules():
    """
    Route for managing schedules:
    - GET: Fetches schedules for the current user.
    - POST: Adds a new schedule.
    - DELETE: Deletes an existing schedule.
    """
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        with get_db() as conn:
            c = conn.cursor()

            # Get user ID from database
            c.execute('SELECT id FROM users WHERE username = ?', (session['username'],))
            user_row = c.fetchone()
            if not user_row:
                return jsonify({'error': 'User not found'}), 404

            user_id = user_row[0]

            if request.method == 'GET':
                # Fetch schedules for the current user
                c.execute('''SELECT id, title, note, schedule_date, schedule_time 
                             FROM schedules 
                             WHERE user_id = ? 
                             ORDER BY schedule_date, schedule_time''', (user_id,))
                schedules = [
                    {'id': row[0], 'title': row[1], 'note': row[2], 
                     'date': row[3], 'time': row[4]}
                    for row in c.fetchall()
                ]
                return jsonify(schedules)

            elif request.method == 'POST':
                # Add a new schedule
                data = request.json

                # Validate required fields
                if not data.get('title'):
                    return jsonify({'error': 'Title is required'}), 400

                # Insert new schedule
                c.execute('''INSERT INTO schedules 
                             (user_id, title, note, schedule_date, schedule_time)
                             VALUES (?, ?, ?, ?, ?)''', 
                             (user_id, data['title'], data.get('note', ''), 
                              data.get('date', ''), data.get('time', '')))
                conn.commit()
                new_id = c.lastrowid  # Get auto-generated ID
                socketio.emit('update_schedule_display', namespace='/host')

                return jsonify({'success': True, 'id': new_id})

            elif request.method == 'DELETE':
                # Delete an existing schedule
                schedule_id = request.args.get('schedule_id', type=int)
                if not schedule_id:
                    return jsonify({'error': 'Missing schedule_id'}), 400

                # Delete only schedules belonging to current user
                c.execute('''DELETE FROM schedules 
                             WHERE id = ? AND user_id = ?''', 
                             (schedule_id, user_id))
                conn.commit()
                socketio.emit('update_schedule_display', namespace='/host')

                return jsonify({'success': True})

    except Exception as e:
        traceback.print_exc()  # Log the full traceback for debugging
        return jsonify({'error': str(e)}), 500
    

# ==== Location Services ====
@app.route('/location', methods=['GET', 'POST'])
def manage_location():
    """
    Unified route for setting and getting the current location.
    
    POST: Sets the current location.
    GET: Retrieves the current location.
    """
    if request.method == 'POST':
        # Set the current location
        data = request.get_json()
        location = data.get('location')
        
        if not location:
            raise BadRequest('Missing location')
        
        # Write the location to the text file
        with open('static/current_location.txt', 'w') as f:
            f.write(location)
        
        socketio.emit('mobile_set_location', namespace='/host')
        
        return jsonify({'success': True, 'location': location})
    
    elif request.method == 'GET':
        # Get the current location
        try:
            with open('static/current_location.txt', 'r') as f:
                location = f.read().strip()
                print(f"Read location: {location}")  # Debugging log
                return jsonify({'location': location})
        except FileNotFoundError:
            print("Location file not found. Using default location: Makkah")  # Debugging log
            return jsonify({'location': 'Makkah al Mukarramah'})

# ==== QR Code Generation ====
@app.route('/qrcode')
def get_qrcode():
    """Generate QR code for current IP address"""
    # Get local IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    qr_url = f'http://{ip}:5000'  # Construct access URL
    qr = qrcode.QRCode(version=1, box_size=5, border=2)
    qr.add_data(qr_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white')
    # Save image to in-memory buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)  # Rewind buffer to beginning
    return send_file(buffer, mimetype='image/png')

    
# ==== Application Startup ====
if __name__ == '__main__':
    # Run on all network interfaces with debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)
