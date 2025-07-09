from classifiers.shoplifting_classifier import STGCN, BATCH_SIZE
from dataset_loaders.ShopLiftingDataset import ShopLiftingDataset
import cv2
import numpy as np

from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

HEADPOINT = "nose"
CAPTURE_SOURCE = "test.mp4"  # Change this to the index of your webcam or video file path
MODEL_LOAD_PATH = "weights/finished/shoplifting.pth"
SHOW_KEYPOINTS = False  # Set to False to disable keypoint visualization

TIMESTEPS = 30  # Number of timesteps for the STGCN model

KP_IM_WIDTH = 512
KP_IM_HEIGHT = 512

model = STGCN(weights_path=MODEL_LOAD_PATH, timesteps=TIMESTEPS).classifier.to(device)

keypoints_arr = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

connections = [
            ('nose', 'left_eye'),
            ('nose', 'right_eye'),
            ('left_eye', 'left_ear'),
            ('right_eye', 'right_ear'),
            ('left_ear', 'left_shoulder'),
            ('right_ear', 'right_shoulder'),
            ('right_shoulder', 'left_shoulder'),
            

            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),

            
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),

            ('right_shoulder', 'right_hip'),
            ('left_shoulder', 'left_hip'),
            ('right_hip', 'left_hip'),

            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),

            
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),

        ]
kp_dict = {}
for i, s in enumerate(keypoints_arr):
    kp_dict[s] = i



def remove_unnecessary_info(kp):
    # Input: shape (T, K, 3) where T is the number of frames, K is the number of keypoints, and 3 is (i, k, x, y)
    new_kp = np.zeros((len(kp), len(kp[0]), 2), dtype=np.float32)
    for i in range(len(kp)):
        for j in range(len(kp[i])):
            new_kp[i,j,:] = kp[i][j][2:]
    return new_kp

def _dim(l, check_for_error):
    if type(l) != list and type(l) != np.ndarray:
        return []
    else:
        if len(l) == 0:
            return [0]
        if type(l[0]) == list and check_for_error:
            next_dim = len(l[0])
            for mini_l in l[1:]:
                if len(mini_l) != next_dim:
                    raise ValueError("Array is sparse")
        return [len(l)] + _dim(l[0], check_for_error)

def dim(l, check_for_error=False):
    return tuple(_dim(l, check_for_error))


def annotate_frame(frame, keypoints, im_height=480, im_width=640):
    # Keypoints is a list of lists, where each index is a keypoint
    x_ind = 0 if len(keypoints[0]) < 4 else 2
    y_ind = 1 if len(keypoints[0]) < 4 else 3
    ann_frame = frame.copy()
    if keypoints is None or len(keypoints) == 0:
        return ann_frame

    for keypoint in keypoints:
        x, y = keypoint[x_ind], keypoint[y_ind]
        if x < 0 or y < 0:
            continue
        if x > ann_frame.shape[1] or y > ann_frame.shape[0]:
            continue
        cv2.circle(ann_frame, (int(x * im_width), int(y*im_height)), 5, (0, 255, 0), -1)
        if x_ind != 0:
            cv2.putText(ann_frame, keypoint[1], (int(x* im_width), int(y * im_height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for conn in connections:
        start_kp, end_kp = kp_dict[conn[0]], kp_dict[conn[1]]
        cv2.line(ann_frame, 
                 (int(keypoints[start_kp][x_ind] * im_width), int(keypoints[start_kp][y_ind] * im_height)),
                 (int(keypoints[end_kp][x_ind] * im_width), int(keypoints[end_kp][y_ind] * im_height)),
                 (255, 0, 0), 2)
    return ann_frame


def draw_box(frame, box, c):
    """
    Draws bounding boxes on the frame.
    
    Args:
        frame (numpy.ndarray): The image frame on which to draw the boxes.
        boxes (list): A list of bounding boxes, where each box is a tuple (x1, y1, x2, y2).
        c (tuple): Color for the bounding box in BGR format.
    """
    c = (c[2], c[1], c[0])
    h, w, _ = frame.shape
    x1, y1, x2, y2 = box
    x1 = int(x1 * w)
    x2 = int(x2 * w)
    y1 = int(y1 * h)
    y2 = int(y2 * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)


def get_available_cameras():
    """
    Tests camera indices to find available webcams.
    Returns a list of indices for cameras that successfully open.
    """
    available_cameras = []
    # Test indices from 0 up to a reasonable number (e.g., 10)
    # Most systems will have cameras at 0, 1, etc.
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            available_cameras.append(i)
            cap.release() # Release the camera immediately after checking
        else:
            # print(f"No camera at index {i}")
            pass # Keep it quiet if no camera is found
    return available_cameras

def display_webcam_feed():
    """
    Captures video from the webcam and displays it in a window.
    Press 'q' to quit the application.
    """
    # Open the default webcam (usually index 0)
    # If you have multiple webcams, you might need to try different indices (1, 2, etc.)
    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    
    # cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, KP_IM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, KP_IM_HEIGHT)

    person_data = ShopLiftingDataset(directory=None, num_timesteps=TIMESTEPS)


    # Check if the webcam was opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Attempting to find available cameras...")
        cameras = get_available_cameras()
        if cameras:
            print(f"Found cameras at indices: {cameras}. Try setting cap = cv2.VideoCapture(index) with one of these.")
        else:
            print("No cameras found. Please ensure your webcam is connected and recognized by the system.")
        return

    print("Webcam feed opened successfully. Press 'q' to quit.")

    frame_count = 0
    frames = []
    tracking_individuals = []
    while True:
        # Read a frame from the webcam
        # ret (boolean): True if the frame was read successfully, False otherwise
        # frame (numpy.ndarray): The captured frame (image)
        ret, frame = cap.read()
        if not ret:
            print("Video Finished")
            break
        person_data.add_frame(frame)
            
        if person_data.has_people():
            print("HAS PEOPLE")
            with torch.no_grad():
                pred = model(person_data.get_skeletons().to(device))[:,-1,:].to("cpu")
            
            for i in range(len(person_data.bboxes)):
                if i >= pred.size()[0]:
                    draw_box(frame, person_data.bboxes[i], (0, 0, 255))

                elif pred[i, 0] < pred[i, 1]:
                    draw_box(frame, person_data.bboxes[i], (255, 0, 0))
                    print(f"Shoplifting detected in individual {i} with confidence {pred[i, 0].item()}")
                    # quit()
                else:
                    draw_box(frame, person_data.bboxes[i], (0, 255, 0))
                    print(f"No Shoplifting detected in individual {i} with confidence {pred[i, 0].item()} and {pred[i, 1].item()}")
        else:
            for i in range(len(person_data.bboxes)):
                draw_box(frame, person_data.bboxes[i], (0, 0, 255))
        # If frame is not read correctly, ret will be False
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the captured frame in a window named 'Webcam Feed'
        cv2.imshow('Webcam Feed', frame)

        # Wait for 1 millisecond and check if the 'q' key is pressed
        # If 'q' is pressed, break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam resource
    cap.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
    print("Webcam feed closed.")

if __name__ == "__main__":
    display_webcam_feed()