from ultralytics import YOLO
import cv2
from playsound import playsound

# Load a pretrained best.pt model
model = YOLO('best.pt')

# Open a video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Run inference on the frame
    results = model(frame,  stream=False, show=True)

    # Check if a helmet is detected
    # helmet_detected = any(r.boxes.cls[0] == 1 for r in results)
    # Check if a helmet is detected
    helmet_detected = any(len(r.boxes.cls) > 0 and r.boxes.cls[0] == 1 for r in results)
    if helmet_detected:
        # Play a buzzer sound (replace 'buzzer_sound.mp3' with your sound file)
        playsound('alarm_buzzer.mp3')

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()