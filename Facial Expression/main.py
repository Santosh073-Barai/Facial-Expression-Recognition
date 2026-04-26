import cv2
import numpy as np

def main():
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    smile_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
    
    if face_cascade.empty() or smile_cascade.empty():
        print("Error: Could not load Haar cascades from OpenCV.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Starting Simulative Emotional Expression System (Fast Heuristic Mode)...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            roi_gray = gray[y:y + h, x:x + w]
            
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
            
            dominant_emotion = "Happy" if len(smiles) > 0 else "Neutral"
            
            text = f"Emotion: {dominant_emotion}"
            color = (0, 215, 255) if dominant_emotion == "Happy" else (200, 200, 200)
            
            cv2.putText(
                frame, text, (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
            
            emoji = ":)" if dominant_emotion == "Happy" else ":|"
            cv2.putText(
                frame, f"Simulated: {emoji}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
            )

        cv2.imshow('Simulative Emotional Expression System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
