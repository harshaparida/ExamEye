from flask import Flask, render_template, request, redirect, flash, url_for, session, jsonify
import mysql.connector
import os
import cv2
import imutils
from flask import Response
import time
from detect_head_movement import initialize_face_mesh as init_head_pose, process_frame as process_head_pose
from detectPhone import load_model as load_yolo, perform_detection as detect_cell_phone, process_results as process_cell_phone
from mouth_detection_movement import initialize_face_mesh as init_mouth_detection, process_image as process_mouth_image, get_landmarks, detect_mouth_opening, draw_results as draw_mouth_results
from detectPerson import load_model as load_ssd, initialize_classes_and_colors, preprocess_frame, detect_objects, process_detections
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/Online_Proctoring_System'
db = SQLAlchemy(app)


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set the upload folder (path to static folder)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Connect to the database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="Online_Proctoring_System"
    )

# Home route for login
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        # Check if the user is an admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)  # Fetch result as dictionary to access columns by name

        cursor.execute("SELECT * FROM admin_credentials WHERE username=%s AND password=%s", (username, password))
        admin = cursor.fetchone()

        if admin:
            # Store the necessary admin info in the session
            session['admin_id'] = admin['admin_id']
            session['admin_username'] = admin['username']

            flash("Admin login successful!")
            return redirect("/admin")
        else:
            # Query the database to verify the username and password for regular users
            cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            user = cursor.fetchone()

            if user:
                # Store the necessary user info in the session
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['full_name'] = user['full_name']
                session['photo'] = user['photo_filename']  # Store photo filename for dashboard

                flash("Login successful!")
                return redirect("/dashboard")
            else:
                flash("Invalid username or password")

        cursor.close()
        conn.close()

    return render_template("login.html")

# Route for admin page
@app.route("/admin")
def admin_page():
    if 'admin_id' in session:  # Check if admin is logged in
        return render_template('admin.html', headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})
    else:
        return redirect(url_for('login'))

# ... rest of the code remains the same ...

# Route for signup page
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form['password']
        photo = request.files['photo']

        # Check if a photo is uploaded
        if not photo:
            flash("Please upload a photo")
            return redirect("/signup")

        # Save the photo to the static/uploads directory
        photo_filename = username + "_" + photo.filename
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
        photo.save(photo_path)

        # Insert user data into the database with the photo filename
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = "INSERT INTO users (full_name, username, password, photo_filename) VALUES (%s, %s, %s, %s)"
        values = (full_name, username, password, photo_filename)

        try:
            cursor.execute(sql, values)
            conn.commit()
            flash("Sign up successful! You can now log in.")
        except mysql.connector.Error as err:
            flash(f"Error: {err}")
        finally:
            cursor.close()
            conn.close()

        return redirect("/")

    return render_template("signup.html")

# Dashboard route
@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:  # Check if user is logged in
        full_name = session['full_name']
        photo = session['photo']  # File name of the profile picture
        photo_url = url_for('static', filename='uploads/' + photo)  # Build the full path to the photo

        return render_template('student.html', full_name=full_name, photo=photo_url, headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})

    else:
        return redirect(url_for('login'))

# Route to logout
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/exam')
def exam():
    # Your exam route code here
    return render_template('exam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    # Initialize all models and required objects
    face_mesh_head_pose = init_head_pose()
    yolo_model = load_yolo()
    face_mesh_mouth = init_mouth_detection()
    ssd_net = load_ssd('models/MobileNetSSD_deploy.prototxt', 'models/MobileNetSSD_deploy.caffemodel')
    classes, colors = initialize_classes_and_colors()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Parameters for mouth opening detection
    threshold = 20
    cheating_duration_threshold = 2.0
    cheating_frequency_threshold = 3
    cheating_period = 10
    mouth_open_start_time = None
    mouth_open_times = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Head Pose Estimation
        frame = process_head_pose(frame, face_mesh_head_pose)

        # Cell Phone Detection
        cell_phone_results = detect_cell_phone(frame, yolo_model)
        frame = process_cell_phone(frame, cell_phone_results)

        # Mouth Opening Detection
        if isinstance(frame, tuple):  # Check if frame is a tuple
            frame = frame[0]  # If it's a tuple, use the first element
        mouth_results, _ = process_mouth_image(frame, face_mesh_mouth)
        if mouth_results.multi_face_landmarks:
            for face_landmarks in mouth_results.multi_face_landmarks:
                landmarks_px = get_landmarks(face_landmarks, frame.shape)
                current_time = time.time()
                cheating_detected, mouth_open_start_time, mouth_open_times = detect_mouth_opening(
                    landmarks_px, threshold, current_time, mouth_open_start_time, mouth_open_times,
                    cheating_duration_threshold, cheating_period
                )
                frame = draw_mouth_results(frame, cheating_detected, mouth_open_times, cheating_frequency_threshold)

        # Webcam Object Detection
        blob, h, w = preprocess_frame(frame)
        detected_objects = detect_objects(ssd_net, blob)
        frame = process_detections(frame, detected_objects, classes, colors, 0.2, h, w)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# admin part

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200), nullable=False)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'))
    question = db.relationship('Question', backref=db.backref('answers', lazy=True))
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'))
    student = db.relationship('Student', backref=db.backref('answers', lazy=True))
    text = db.Column(db.String(200), nullable=False)

@app.route('/upload_question', methods=['GET', 'POST'])
def upload_question():
    if request.method == 'POST':
        question = Question(text=request.form['question'])
        db.session.add(question)
        db.session.commit()
        return redirect(url_for('view_questions'))
    return render_template('upload_question.html')

@app.route('/view_questions')
def view_questions():
    questions = Question.query.all()
    return render_template('view_questions.html', questions=questions)

@app.route('/reset_questions')
def reset_questions():
    Question.query.delete()
    db.session.commit()
    return redirect(url_for('view_questions'))

@app.route('/view_students')
def view_students():
    students = Student.query.all()
    return render_template('view_students.html', students=students)






if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
