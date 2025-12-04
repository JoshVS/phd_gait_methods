import cv2

# Force the V4L2 backend
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# This is where the select() timeout would happen
ret, frame = cap.read()

if not ret:
    print("Error: Can't receive frame. The camera is connected, but no data is being sent.")
    print("This is often due to a driver issue or a problem with the usbipd stream.")
else:
    print(f"Success! A frame was captured with shape: {frame.shape}")

cap.release()