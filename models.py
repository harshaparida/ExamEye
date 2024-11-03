from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class CheatingEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)

    def __repr__(self):
        return f'<CheatingEvent {self.id} at {self.timestamp}>'