from flask import Flask, render_template, request, redirect, flash, url_for, session, jsonify, stream_with_context, \
    send_from_directory
import mysql.connector
import os
import cv2
import imutils
from flask import Response
import time
from detect_head_movement import initialize_face_mesh as init_head_pose, process_frame as process_head_pose
from detectPhone import load_model as load_yolo, perform_detection as detect_cell_phone, process_results as process_cell_phone
from mouth_detection_movement import initialize_face_mesh as init_mouth_detection, process_image as process_mouth_image, get_landmarks, detect_mouth_opening, draw_results as draw_mouth_results
from flask import Flask, render_template, request, redirect, url_for
from detectPerson import load_model, initialize_classes_and_colors, preprocess_frame, detect_objects, process_detections
import pymysql
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16 MB
app.secret_key = 'your_secret_key'

# Set the upload folder (path to static folder)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_FOLDER = 'static/recordings'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Connect to the database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="online_proctoring_system"
    )

# Home route for login
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        # Connect to the database
        conn = get_db_connection()

        # Check for admin login
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM admin_credentials WHERE username=%s AND password=%s",
                       (username, password))
        admin = cursor.fetchone()

        if admin:
            session['admin_id'] = admin['admin_id']
            session['admin_username'] = admin['username']
            flash("Admin login successful!")
            cursor.close()
            conn.close()
            return redirect("/admin")

        # Close the first cursor before opening a new one
        cursor.close()

        # Check for user login
        cursor = conn.cursor(dictionary=True)  # New cursor for the user query
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s",
                       (username, password))
        user = cursor.fetchone()

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['photo'] = user['photo_filename']
            session.permanent = True

            flash("Login successful!")
            cursor.close()
            conn.close()
            return redirect("/dashboard")

        # If login fails
        flash("Invalid username or password")

        # Close the cursor and connection after use
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


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


cheating_detected = False
@app.route('/cheating_alert')
def cheating_alert():
    def generate():
        global cheating_detected
        while True:
            if cheating_detected:
                yield f"data: Cheating detected!\n\n"  # The message format for EventSource
                cheating_detected = False  # Reset the flag after sending the alert
            time.sleep(1)  # Check every second

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/check_cheating', methods=['GET'])
def check_cheating():
    return jsonify({'cheatingDetected': cheating_detected})

def insert_cheating_event(student_id, event_type):
    connection = get_db_connection()
    cursor = connection.cursor()
    sql = "INSERT INTO cheating_events (student_id, event_type) VALUES (%s, %s)"
    cursor.execute(sql, (student_id, event_type))
    connection.commit()
    cursor.close()
    connection.close()

def gen_frames():
    global cheating_detected

    # Initialize all models and required objects
    face_mesh_head_pose = init_head_pose()
    yolo_model = load_model()  # Load YOLOv5 model
    face_mesh_mouth = init_mouth_detection()
    classes = initialize_classes_and_colors()  # Initialize classes for YOLOv5

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Parameters for mouth opening detection
    threshold = 20
    cheating_duration_threshold = 2.0
    cheating_frequency_threshold = 3
    cheating_period = 10
    mouth_open_start_time = None
    mouth_open_times = []

    last_alert_time = 0
    alert_cooldown = 5  # seconds

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Initialize the cheating flag
        cheating_detected = False

        # Head Pose Estimation
        frame = process_head_pose(frame, face_mesh_head_pose)
        head_pose_detected = False  # Update based on your detection logic

        # Check for head pose cheating
        if head_pose_detected:
            cheating_detected = True
            print("Alert: Suspicious head movement detected!")
            insert_cheating_event(student_id=1, event_type='Head Pose Detected')  # Replace with actual student_id

        # Cell Phone Detection (using YOLOv5)
        cell_phone_results = detect_cell_phone(frame, yolo_model)
        frame = process_cell_phone(frame, cell_phone_results)
        if cell_phone_results:  # Check if results are not empty
            cheating_detected = True
            insert_cheating_event(student_id=1, event_type='Cell Phone Detected')  # Replace with actual student_id

        # Mouth Opening Detection
        if isinstance(frame, tuple):
            frame = frame[0]
        mouth_results, _ = process_mouth_image(frame, face_mesh_mouth)
        if mouth_results.multi_face_landmarks:
            for face_landmarks in mouth_results.multi_face_landmarks:
                landmarks_px = get_landmarks(face_landmarks, frame.shape)
                current_time = time.time()
                cheating_detected, mouth_open_start_time, mouth_open_times = detect_mouth_opening(
                    landmarks_px, threshold, current_time, mouth_open_start_time, mouth_open_times,
                    cheating_duration_threshold, cheating_period
                )
                if cheating_detected:
                    cheating_detected = True  # Set the flag when cheating is detected
                    insert_cheating_event(student_id=1, event_type='Mouth Open Detected')  # Replace with actual student_id
                frame = draw_mouth_results(frame, cheating_detected, mouth_open_times, cheating_frequency_threshold)

        # Webcam Object Detection with YOLOv5
        results = detect_objects(yolo_model, frame)
        frame = process_detections(frame, results, classes)

        # Yield the processed frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Frame not encoded.")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/cheating_events')
def cheating_events():
    cur = mysql.connector.cursor()  # Create a cursor object
    cur.execute("SELECT * FROM cheating_events")  # Execute the SQL query
    results = cur.fetchall()  # Fetch all results
    cur.close()  # Close the cursor

    return render_template('cheating_events.html', events=results)

# admin part

@app.route('/upload_question', methods=['GET', 'POST'])
def upload_question():
    if request.method == 'POST':
        # Get the question from the form
        question = request.form['question']

        # Get a database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert the question into the database
        query = "INSERT INTO questions (text) VALUES (%s)"
        cursor.execute(query, (question,))
        conn.commit()

        # Close the database connection
        cursor.close()
        conn.close()


    return render_template('upload_question.html')

@app.route('/view_questions')
def view_questions():
    # Get a database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    # Execute the query to get all questions
    query = "SELECT * FROM questions"
    cursor.execute(query)
    questions = cursor.fetchall()

    # Close the database connection
    cursor.close()
    conn.close()

    # Return the questions
    return render_template('view_question.html', questions=questions)

@app.route('/reset_questions')
def reset_questions():
    # Get a database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    # Execute the query to delete all questions
    query = "DELETE FROM questions"
    cursor.execute(query)
    conn.commit()

    # Close the database connection
    cursor.close()
    conn.close()

    # Redirect to the view questions page
    return redirect(url_for('view_questions'))

@app.route('/view_students')
def view_students():
    # Get a database connection
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)  # Use dictionary cursor to get column names

    # Execute the query to get all students
    query = "SELECT id, username, photo_filename FROM users"
    cursor.execute(query)
    students = cursor.fetchall()

    # Close the database connection
    cursor.close()
    conn.close()

    # Pass the students to the HTML template
    return render_template('view_student.html', students=students)

@app.route('/admin/view_report/<int:student_id>')
def view_report(student_id):
    try:
        cnx = get_db_connection()
        cursor = cnx.cursor()

        # Query to fetch answers
        answers_query = """
            SELECT q.text, a.text 
            FROM answers a 
            JOIN questions q ON a.question_id = q.id 
            WHERE a.student_id = %s
        """
        cursor.execute(answers_query, (student_id,))
        answers = cursor.fetchall()

        # Query to fetch cheating events
        cheating_events_query = """
            SELECT event_type, timestamp 
            FROM cheating_events 
            WHERE student_id = %s
        """
        cursor.execute(cheating_events_query, (student_id,))
        cheating_events = cursor.fetchall()

        return render_template('view_report.html', student_id=student_id, answers=answers,
                               cheating_events=cheating_events)

    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
        return "Error: Unable to retrieve answers", 500
    finally:
        cursor.close()
        cnx.close()



@app.route('/exam', methods=['GET', 'POST'])
def exam():
    if request.method == 'POST':
        # Get a database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get the answers from the form
        answers = request.form

        # Insert the answers into the database
        for question_id, answer in answers.items():
            if question_id != 'csrf_token':  # Ignore the CSRF token
                query = "INSERT INTO answers (question_id, student_id, text) VALUES (%s, %s, %s)"
                cursor.execute(query, (question_id, session['user_id'], answer))
                conn.commit()

        # Close the database connection
        cursor.close()
        conn.close()

        # Return a success message
        flash('Answers submitted successfully!')
        return redirect(url_for('dashboard'))
    else:
        # Get a database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Execute the query to get all questions
        query = "SELECT * FROM questions"
        cursor.execute(query)
        questions = cursor.fetchall()

        # Retrieve the username from the database
        username = "Guest"  # Default username
        if 'user_id' in session:
            cursor.execute("SELECT username FROM users WHERE id = %s", (session['user_id'],))
            user_result = cursor.fetchone()
            if user_result:
                username = user_result[0]  # Get the username from the result

        # Close the database connection
        cursor.close()
        conn.close()

        # Return the questions and username to the template
        return render_template('exam.html', questions=questions, username=username)

@app.route('/upload_recording', methods=['POST'])
def upload_recording():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"File saved to {file_path}")  # Log the file path
        return jsonify({"message": "File uploaded successfully"}), 200
    return jsonify({"error": "Failed to upload file"}), 400

# List recordings for admin
@app.route('/admin/recordings', methods=['GET'])
def list_recordings():
    # Get the list of recordings from the folder
    recordings = os.listdir(UPLOAD_FOLDER)
    return render_template('recordings.html', recordings=recordings)

# Serve recordings directly
@app.route('/recordings/<filename>', methods=['GET'])
def serve_recording(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
