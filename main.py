import cv2
from deepface import DeepFace

# Load face cascade for detection
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Read and analyze faces
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture image")
        break
    
    try:
        # Analyze the face using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'])  # Only analyze emotion
        if isinstance(result, list):  # Check if result is a list
            dominant_emotion = result[0]['dominant_emotion']  # Access the first face's emotion
        else:
            dominant_emotion = result['dominant_emotion']     # Access directly if it's a dict
    except Exception as e:
        print("Error during face analysis:", e)
        dominant_emotion = "No Face Detected"

    # Detect faces and draw rectangles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add emotion text to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, dominant_emotion, (10, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame,"Enter q to exit", (10, 70), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Demo Video', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
