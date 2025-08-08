DROP TABLE IF EXISTS assignments;

CREATE TABLE assignments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    course_id VARCHAR(255) NOT NULL,
    course_title VARCHAR(255) NOT NULL,
    instructor_name VARCHAR(255) NOT NULL,
    pdf_link VARCHAR(512) NOT NULL,
    topic VARCHAR(100) NOT NULL
);

INSERT INTO assignments (course_id, course_title, instructor_name, pdf_link, topic) VALUES
('PEF301', 'Power Electronics Lab', 'Dr. G. Rao', '.\\mysql-init\\pdfs\\Assignment 1 - Mechatronics.pdf', 'Boost and Buck Converter Design'),
('CSD345', 'Mobile Application Development', 'Dr. D. Ramesh', '.\\mysql-init\\pdfs\\Lab_Assignment-1 - Embedded System Design.pdf', 'SQLite in Flutter');
