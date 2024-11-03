from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)  # Use the first camera

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a selfie-view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        pose = "Looking Center"  # Default pose

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)

                # Get the 3D coordinates of the nose tip
                nose = face_landmarks.landmark[mp_face_mesh.FACEMESH_NOSE_TIP]
                nose_x = int(nose.x * frame.shape[1])
                nose_y = int(nose.y * frame.shape[0])

                # Determine head pose based on nose position
                if nose_x < frame.shape[1] // 3:
                    pose = "Looking Left"
                elif nose_x > frame.shape[1] * 2 // 3:
                    pose = "Looking Right"
                elif nose_y < frame.shape[0] // 3:
                    pose = "Looking Up"
                elif nose_y > frame.shape[0] * 2 // 3:
                    pose = "Looking Down"

                # Display the pose on the frame
                cv2.putText(frame, pose, (nose_x, nose_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose')
def get_pose():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    if not success:
        return jsonify({'pose': 'Error'})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    pose = "Looking Center"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the nose tip landmark (index 1)
            nose = face_landmarks.landmark[1]  # Index for the nose tip
            nose_x = int(nose.x * frame.shape[1])
            nose_y = int(nose.y * frame.shape[0])

            if nose_x < frame.shape[1] // 3:
                pose = "Looking Left"
            elif nose_x > frame.shape[1] * 2 // 3:
                pose = "Looking Right"
            elif nose_y < frame.shape[0] // 3:
                pose = "Looking Up"
            elif nose_y > frame.shape[0] * 2 // 3:
                pose = "Looking Down"

    cap.release()
    return jsonify({'pose': pose})

if __name__ == '__main__':
    app.run(debug=True)