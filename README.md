import cv2
import numpy as np
import winsound

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire color ranges (red, orange, yellow)
    lower1 = np.array([0, 150, 150])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([15, 100, 100])
    upper2 = np.array([35, 255, 255])

    # Combine masks
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Remove noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:  # Minimum fire area to trigger
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE DETECTED!", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            fire_detected = True

    # Play alarm if fire detected
    if fire_detected:
        winsound.Beep(1000, 500)

    # Show output
    cv2.imshow("Fire Detection", frame)
    cv2.imshow("Fire Mask", mask)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
# Fire-Detection
This project uses OpenCV to detect fire in real-time from a webcam feed by identifying red, orange, and yellow color ranges in the HSV spectrum. It combines multiple color masks, removes noise, and detects irregular shapes using contour analysis to reduce false alarms.
