from flask import Flask, render_template, request, redirect, url_for
import mysql.connector

app = Flask(__name__)

# MySQL database connection settings
username = 'root'
password = 'rootroot'
host = 'localhost'
database = 'your_database'

# Connect to MySQL database
cnx = mysql.connector.connect(
    user='root',
    password='rootroot',
    host='localhost',
    database='your_database',
    auth_plugin='mysql_native_password'
)
# Define the connection settings as a dictionary
cursor = cnx.cursor()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        if user:
            return redirect(url_for('success'))
        else:
            return 'Invalid username or password', 401
    return render_template('index.html')

@app.route('/success')
def success():
    return 'Login successful!'

if __name__ == '__main__':
    app.run(debug=True)