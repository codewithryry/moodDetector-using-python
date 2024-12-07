import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

def is_focused(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Compute the Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and facial expressions
    try:
        # Using DeepFace to analyze facial expressions and age
        result = DeepFace.analyze(frame, actions=['emotion', 'age'], enforce_detection=True)

        # Get the dominant emotion and predicted age
        emotion = result[0]['dominant_emotion']
        age = result[0]['age']

        # Draw a rectangle around the face (face detection)
        for face in result[0]['region']:
            # Accessing coordinates correctly
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the emotion and age on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Age: {age}', (50, 100), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Check the focus of the frame
        focus_var = is_focused(frame)
        if focus_var < 100:  # Threshold for blur detection
            cv2.putText(frame, 'Focus: OUT OF FOCUS', (50, 150), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Focus: IN FOCUS', (50, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print(f"Error detecting emotion and age: {e}")
    
    # Show the resulting frame
    cv2.imshow('Expression Camera', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
