from flask import Flask, render_template, request, redirect, flash, url_for, session
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

        # Query the database to verify the username and password
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)  # Fetch result as dictionary to access columns by name

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

        return render_template('student.html', full_name=full_name, photo=photo_url)
    else:
        return redirect(url_for('login'))

# Route to logout
@app.route("/logout")
def logout():
    session.clear()  # Clear the session on logout
    flash("You have been logged out.")
    return redirect("/")

@app.route('/start-exam')
def start_exam():
    # Code to start the exam goes here
    return render_template('exam.html')

if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
