from flask import Flask, render_template, request, redirect, flash, url_for, session, jsonify
import mysql.connector
import os

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
        database="user_credentials"
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

@app.route('/start-exam')
def start_exam():
    # Code to start the exam goes here
    return render_template('exam.html')

@app.route('/uploadQuestion', methods=['GET', 'POST'])
def uploadQuestion():
    if 'admin_id' in session:  # Check if admin is logged in
        if request.method == 'POST':
            question = request.form['question']
            answer = request.form['answer']

            # Insert question data into the database
            conn = get_db_connection()
            cursor = conn.cursor()

            sql = "INSERT INTO questions (question, answer) VALUES (%s, %s)"
            values = (question, answer)

            try:
                cursor.execute(sql, values)
                conn.commit()
                flash("Question uploaded successfully!")
            except mysql.connector.Error as err:
                flash(f"Error: {err}")
            finally:
                cursor.close()
                conn.close()

            return redirect("/admin")
        else:
            return render_template('uploadQuestion.html')
    else:
        return redirect(url_for('login'))

@app.route('/view_questions')
def view_questions():
    if 'admin_id' in session:  # Check if admin is logged in
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM questions")
        questions = cursor.fetchall()

        return render_template('view_questions.html', questions=questions)
    else:
        return redirect(url_for('login'))

@app.route('/answer_question', methods=['GET', 'POST'])
def answer_question():
    if 'user_id' in session:  # Check if user is logged in
        if request.method == 'POST':
            question_id = request.form['question_id']
            answer = request.form['answer']

            # Check if the answer is correct
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT answer FROM questions WHERE id=%s", (question_id,))
            correct_answer = cursor.fetchone()

            if correct_answer and correct_answer['answer'] == answer:
                flash("Correct answer!")
            else:
                flash("Incorrect answer!")

            return redirect("/dashboard")
        else:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM questions")
            questions = cursor.fetchall()

            return render_template('answer_question.html', questions=questions)
    else:
        return redirect(url_for('login'))

@app.route('/save_answer', methods=['POST'])
def save_answer():
    if 'user_id' in session:  # Check if user is logged in
        student_id = session['user_id']
        question_id = request.form['question_id']
        answer = request.form['answer']

        # Check if the answer is correct
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT answer FROM questions WHERE id=%s", (question_id,))
        correct_answer = cursor.fetchone()

        if correct_answer and correct_answer['answer'] == answer:
            marks = 1
        else:
            marks = 0

        # Save student answer and marks
        cursor.execute("INSERT INTO student_answers (student_id, question_id, answer, marks) VALUES (%s, %s, %s, %s)", (student_id, question_id, answer, marks))
        conn.commit()

        return redirect("/dashboard")
    else:
        return redirect(url_for('login'))

@app.route('/student_answers')
def student_answers():
    if 'admin_id' in session:  # Check if admin is logged in
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users")

        return render_template('student_answers.html')
    else:
        return redirect(url_for('login'))

@app.route('/view_student_answers/<int:student_id>')
def view_student_answers(student_id):
    if 'admin_id' in session:  # Check if admin is logged in
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM student_answers WHERE student_id=%s", (student_id,))
        answers = cursor.fetchall()

        return render_template('view_student_answers.html', answers=answers)
    else:
        return redirect(url_for('login'))

if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
