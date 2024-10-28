DROP DATABASE IF EXISTS Online_Proctoring_System;

-- Create a new database
CREATE DATABASE Online_Proctoring_System;

-- Use the newly created database
USE Online_Proctoring_System;

-- Create the users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    photo LONGBLOB,
    photo_filename VARCHAR(255)  -- Move this into the table definition for clarity
);

-- Create the admin credentials table
CREATE TABLE admin_credentials (
    admin_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- Insert admin credentials
INSERT INTO admin_credentials (username, password) VALUES ('admin', 'password');

-- Create the questions table
CREATE TABLE questions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text VARCHAR(200) NOT NULL
);

-- Create the students table
CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- Create the answers table
drop table answers;
CREATE TABLE answers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question_id INT,
    student_id INT,
    text VARCHAR(200) NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE
);
SET GLOBAL innodb_lock_wait_timeout = 120;
select * from answers;
select * from users; 

CREATE TABLE cheating_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE
);

select * from cheating_events;