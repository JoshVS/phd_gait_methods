from classifiers.shoplifting_classifier import STGCN, BATCH_SIZE
import cv2
import numpy as np

from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

HEADPOINT = "nose"
CAPTURE_SOURCE = "test.mp4"  # Change this to the index of your webcam or video file path
MODEL_LOAD_PATH = "weights/finished/shoplifting.pth"
SHOW_KEYPOINTS = False  # Set to False to disable keypoint visualization

KP_IM_WIDTH = 128
KP_IM_HEIGHT = 128

model = STGCN(weights_path=MODEL_LOAD_PATH).classifier

yolo_model = YOLO("yolov8x-pose-p6.pt")

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

# Thanks Gemini
def make_keypoints_rotation_invariant(keypoints, shoulder_indices=(5, 6), reference_point_index=0):
    """
    Makes a set of 2D keypoints rotation-invariant.

    This function normalizes the rotation of a pose by aligning a reference bone
    (e.g., shoulders) with the horizontal axis. It then translates the pose
    so a specific reference point (e.g., nose) is at the origin.

    Args:
        keypoints (list or np.array): A Python list of lists or a NumPy array of shape (N, 2)
                                      where N is the number of keypoints, and each inner list/row
                                      is an (x, y) coordinate. Ultralytics models typically output
                                      (x, y, confidence) so ensure you pass only the (x, y) part.
        shoulder_indices (tuple): A tuple (left_shoulder_index, right_shoulder_index)
                                  representing the indices of the shoulder keypoints
                                  in the `keypoints` array. Default assumes COCO format
                                  where 5 is left shoulder and 6 is right shoulder.
        reference_point_index (int): The index of the keypoint to use as the
                                     translation reference (e.g., nose). This
                                     point will be moved to (0,0) after rotation.
                                     Default assumes COCO format where 0 is nose.

    Returns:
        np.array: A NumPy array of shape (N, 2) with rotation-invariant keypoints.
                  The keypoints are rotated and translated.
    """
    # Convert input list to NumPy array if it's not already one
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints, dtype=float)

    if keypoints.shape[0] < max(shoulder_indices) + 1:
        print("Warning: Not enough keypoints for specified shoulder indices.")
        # Return original keypoints if reference points are missing
        return keypoints

    # Extract shoulder keypoints
    left_shoulder = keypoints[shoulder_indices[0]]
    right_shoulder = keypoints[shoulder_indices[1]]

    # Calculate the vector from left to right shoulder
    shoulder_vector = right_shoulder - left_shoulder

    # Calculate the angle of the shoulder vector with the positive x-axis
    # atan2 gives the angle in radians, handling all quadrants
    angle_rad = np.arctan2(shoulder_vector[1], shoulder_vector[0])

    # Create a 2D rotation matrix for rotation by -angle_rad
    # This rotates the pose so the shoulder line becomes horizontal
    cos_val = np.cos(-angle_rad)
    sin_val = np.sin(-angle_rad)
    rotation_matrix = np.array([
        [cos_val, -sin_val],
        [sin_val, cos_val]
    ])

    # Translate all keypoints so the left shoulder is at the origin before rotation
    # This ensures rotation happens around a point related to the pose itself
    translated_keypoints = keypoints - left_shoulder

    # Apply the rotation to all keypoints
    rotated_keypoints = np.dot(translated_keypoints, rotation_matrix.T)

    # Now, translate the rotated keypoints so the chosen reference point (e.g., nose)
    # is at the origin (0,0). This makes the pose translation-invariant.
    translation_vector = rotated_keypoints[reference_point_index]
    rotation_invariant_keypoints = rotated_keypoints - translation_vector

    return rotation_invariant_keypoints


def normalize_keypoints(keypoints, im_height=480, im_width=640):
    norm_kp = []
    for i in range(len(keypoints)):
        norm_kp.append(make_keypoints_rotation_invariant(keypoints[i]))
    return norm_kp


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


def im2kp(frame, min_frames = 2, im_height=512, im_width=640):
    torch.set_default_dtype(torch.float32)
    
    
    # bounding_boxes = yolo_model([os.path.join(filename, f) for f in frames], save=False, verbose=False)
    

    found_frame = False
    # print(os.path.join(filename, frame))
    # quit()
    curr_frame = []
    try:
        results_generator = yolo_model.track(source=frame, show=True, conf=0.3, save=False, stream=False, verbose=True, imgsz=(KP_IM_WIDTH, KP_IM_HEIGHT), max_det=1000)
        # print(len(results_generator))
        # quit()
    except Exception as e:
        print(e)
        return None
    for i in range(2):
        for k in kp_dict.keys():
            curr_frame.append([i, k, 0, 0])

    # print(dir(results_generator[0]))
    # quit()
    
    for i, res in enumerate(results_generator):
        if i >= 2:
            break
        
        
        
        kpts = res.keypoints.xy.cpu().numpy()[0, ...] # Shape (17, 2)
        if kpts.shape[0] != 0:
            found_frame = True  
            bbox = res.boxes.xywh.cpu().numpy()[0, ...] # Shape (4,)
            xA, yA, w, h = bbox
            for k in kp_dict.keys():
                v = kp_dict[k]
                curr_frame[kpts.shape[0] * i + v] = [i,
                                                      k, 
                                                      (kpts[v, 0] - xA) / w, 
                                                      (yA - kpts[v, 1]) / h]
    

        # if pose_results.pose_landmarks is not None:
        #     found_frame = True
        #     for k in kp_dict.keys():
        #         v = kp_dict[k]
        #         curr_frame.append([i, k, pose_results.pose_landmarks.landmark[v].x, pose_results.pose_landmarks.landmark[v].y, pose_results.pose_landmarks.landmark[v].z])
                # print(dir(pose_results.pose_world_landmarks))
                # quit()
    if found_frame:
        return curr_frame
    else:
        print("No frame found")
        return None

def reshape_skeletons(skeletons):
        if len(skeletons) %17 != 0:
            raise ValueError("Skeletons must be a multiple of 17 keypoints per person")
        reshaped_skel = [None] * (len(skeletons) // 17)
        for i in range(len(reshaped_skel)):
            reshaped_skel[i] = skeletons[i*17:(i+1)*17]
        return reshaped_skel



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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)


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
        print("Getting Keypoints")
        keypoints = im2kp(frame)
        if keypoints is None:
            print("No keypoints found in the frame.")
            
            print("Displaying Frame")        
            cv2.imshow('Webcam Feed', frame)
            continue
        keypoints = reshape_skeletons(keypoints)
        if len(keypoints) > len(tracking_individuals):
            tracking_individuals.extend([[]] * (len(keypoints) - len(tracking_individuals)))
            
        for p in range(len(keypoints)):
            if len(tracking_individuals[p]) >= 99:
                tracking_individuals[p] = tracking_individuals[p][-99:]
            tracking_individuals[p].append(keypoints[p])
        if len(tracking_individuals) > 0:
            for i in range(len(tracking_individuals)):
                if len(tracking_individuals[i]) == 100:
                    X = remove_unnecessary_info(tracking_individuals[i]) # Shape (T, K, 2)
                    X = normalize_keypoints(X, im_height=frame.shape[0], im_width=frame.shape[1])
                    X = np.array(X, dtype=np.float32)
                    if SHOW_KEYPOINTS:
                        frame = annotate_frame(frame, X[-1])
                    C = 2
                    T = 0
                    K = 1
                    X = X.transpose((C, T, K)) # Shape (2, T, K)
                    X = X.reshape((1,) + X.shape + (1,)) # Shape (1, 2, T, K)
                    X = torch.tensor(X, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        pred = model(X)[0,-1,:]
                        # means = torch.mean(pred, dim=0, keepdim=True)[0,:]
                        if pred[0] < pred[1]:
                            print(f"Shoplifting detected in individual {i} with confidence {pred[0].item()}")
                            quit()
                        
                    print(f"No Shoplifting detected in individual {i} with confidence {pred[0].item()} and {pred[1].item()}")
        print("Got Keypoints")
        print("Annotated Frame")
        # If frame is not read correctly, ret will be False
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the captured frame in a window named 'Webcam Feed'
        print("Displaying Frame")        
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