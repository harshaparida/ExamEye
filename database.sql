CREATE DATABASE user_credentials;
USE user_credentials;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    photo LONGBLOB 
);
ALTER TABLE users ADD photo_filename VARCHAR(255);
drop table admin_credentials;
CREATE TABLE admin_credentials (
  admin_id INT AUTO_INCREMENT,
  username VARCHAR(50) NOT NULL,
  password VARCHAR(255) NOT NULL,
  PRIMARY KEY (admin_id)
);
INSERT INTO admin_credentials (username, password) VALUES ('admin', 'password');

CREATE TABLE questions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    question VARCHAR(200) NOT NULL,
    answer VARCHAR(200) NOT NULL
);

CREATE TABLE student_answers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    question_id INT NOT NULL,
    answer VARCHAR(200) NOT NULL,
    marks INT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (question_id) REFERENCES questions(id)
);

select * from users;
select * from questions;
select * from student_answers;
