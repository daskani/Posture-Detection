import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between two points and the vertical axis
def calculate_angle(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    angle_rad = math.atan2(delta_x, delta_y)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Function to calculate the midpoint between two points
def midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

# Function to check posture correctness and infer possible conditions
def check_posture(left_shoulder_px, right_shoulder_px, left_hip_px, right_hip_px, left_knee_px, right_knee_px, spine_angle):
    posture_corrections = []
    possible_conditions = []

    # Calculate shoulder alignment
    shoulder_alignment = abs(left_shoulder_px[1] - right_shoulder_px[1])
    if shoulder_alignment > 20:  # Threshold for misalignment in pixels
        posture_corrections.append("Shoulder Misalignment")
        if shoulder_alignment > 40:
            possible_conditions.append("Severe Misalignment: Consider Scoliosis or Rotator Cuff Injury")
        else:
            possible_conditions.append("Mild Misalignment: May indicate Poor Posture or Muscle Imbalance")

    # Check hip alignment
    hip_alignment = abs(left_hip_px[1] - right_hip_px[1])
    if hip_alignment > 20:  # Threshold for misalignment in pixels
        posture_corrections.append("Hip Misalignment")
        if hip_alignment > 40:  # Threshold for severe misalignment
            possible_conditions.append("Severe Hip Misalignment: Consider Pelvic Tilt or Leg Length Discrepancy")
        else:
            possible_conditions.append("Mild Hip Misalignment: May indicate Muscle Imbalance")

    # Check knee alignment
    knee_alignment = abs(left_knee_px[1] - right_knee_px[1])
    if knee_alignment > 20:  # Threshold for misalignment in pixels
        posture_corrections.append("Knee Misalignment")
        possible_conditions.append("Knee Misalignment: May indicate Joint Issues or Gait Problems")

    # Check spinal angle for overall posture
    if abs(spine_angle) > 5:  # Threshold for spinal alignment issues
        posture_corrections.append("Spinal Misalignment")
        if abs(spine_angle) > 15:
            possible_conditions.append("Severe Spinal Misalignment: Consider Scoliosis")
        else:
            possible_conditions.append("Mild Spinal Misalignment: May indicate Poor Posture")

    # Check overall posture
    if len(posture_corrections) == 0:
        return "Good Posture", "No obvious conditions detected"
    else:
        return ", ".join(posture_corrections), "; ".join(possible_conditions)

# Function to determine activity and posture based on landmarks
def detect_activity_and_posture(landmarks, frame_width, frame_height):
    # Extract relevant landmarks
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    except KeyError:
        return "Unrecognized", "improper placement of the camera"

    # Convert normalized coordinates to pixel values
    def to_pixel_coords(landmark):
        return (landmark.x * frame_width, landmark.y * frame_height)

    left_shoulder_px = to_pixel_coords(left_shoulder)
    right_shoulder_px = to_pixel_coords(right_shoulder)
    left_elbow_px = to_pixel_coords(left_elbow)
    right_elbow_px = to_pixel_coords(right_elbow)
    left_wrist_px = to_pixel_coords(left_wrist)
    right_wrist_px = to_pixel_coords(right_wrist)
    left_hip_px = to_pixel_coords(left_hip)
    right_hip_px = to_pixel_coords(right_hip)
    left_knee_px = to_pixel_coords(left_knee)
    right_knee_px = to_pixel_coords(right_knee)
    left_ankle_px = to_pixel_coords(left_ankle)
    right_ankle_px = to_pixel_coords(right_ankle)
    nose_px = to_pixel_coords(nose)

    # Calculate key distances and angles
    shoulder_width = calculate_distance(left_shoulder_px, right_shoulder_px)
    hip_width = calculate_distance(left_hip_px, right_hip_px)
    arm_left_angle = calculate_angle(left_shoulder_px, left_elbow_px)
    arm_right_angle = calculate_angle(right_shoulder_px, right_elbow_px)
    knee_left_angle = calculate_angle(left_hip_px, left_knee_px)
    knee_right_angle = calculate_angle(right_hip_px, right_knee_px)
    spine_angle = calculate_angle(nose_px, midpoint(left_hip_px, right_hip_px))  # Approximation for spinal alignment

    # Determine activity based on landmark positions
    if abs(left_ankle_px[1] - right_ankle_px[1]) < 0.2 * frame_height and abs(left_knee_px[1] - right_knee_px[1]) < 0.2 * frame_height:
        if abs(left_elbow_px[0] - left_wrist_px[0]) < 0.2 * frame_width and abs(right_elbow_px[0] - right_wrist_px[0]) < 0.2 * frame_width:
            if abs(left_hip_px[1] - right_hip_px[1]) < 0.2 * frame_height and abs(left_shoulder_px[1] - right_shoulder_px[1]) < 0.2 * frame_height:
                if abs(nose_px[1] - (midpoint(left_shoulder_px, right_shoulder_px)[1])) < 0.2 * frame_height:
                    return "Sitting", check_posture(left_shoulder_px, right_shoulder_px, left_hip_px, right_hip_px, left_knee_px, right_knee_px, spine_angle)
                else:
                    return "Reading", check_posture(left_shoulder_px, right_shoulder_px, left_hip_px, right_hip_px, left_knee_px, right_knee_px, spine_angle)
        elif calculate_distance(left_wrist_px, nose_px) < shoulder_width or calculate_distance(right_wrist_px, nose_px) < shoulder_width:
            return "Eating", check_posture(left_shoulder_px, right_shoulder_px, left_hip_px, right_hip_px, left_knee_px, right_knee_px, spine_angle)
    elif abs(left_ankle_px[1] - right_ankle_px[1]) > 0.2 * frame_height and abs(left_knee_px[1] - right_knee_px[1]) > 0.2 * frame_height:
        if abs(left_shoulder_px[1] - right_shoulder_px[1]) > 0.2 * frame_height or abs(left_hip_px[1] - right_hip_px[1]) > 0.2 * frame_height:
            return "Walking", "Check if the stride is even"

    return "Unrecognized", "May be Poor Posture"


# Function to check posture correctness and infer possible conditions
def check_posture(left_shoulder_px, right_shoulder_px, left_hip_px, right_hip_px, left_knee_px, right_knee_px, spine_angle):
    posture_corrections = []
    possible_conditions = []

    # Calculate shoulder alignment
    shoulder_alignment = abs(left_shoulder_px[1] - right_shoulder_px[1])

    # Check for shoulder misalignment
    if shoulder_alignment > 20:  # Threshold for misalignment in pixels
        posture_corrections.append("Shoulder Misalignment")

        if shoulder_alignment > 40:  # Threshold for severe misalignment
            possible_conditions.append("Severe Misalignment: Consider Scoliosis or Rotator Cuff Injury")
        else:
            possible_conditions.append("Mild Misalignment: May indicate Poor Posture or Muscle Imbalance")

        # Check hip alignment
        hip_alignment = abs(left_hip_px[1] - right_hip_px[1])
        if hip_alignment > 20:  # Threshold for misalignment in pixels
            posture_corrections.append("Hip Misalignment")
            if hip_alignment > 40:  # Threshold for severe misalignment
                possible_conditions.append("Severe Hip Misalignment: Consider Pelvic Tilt or Leg Length Discrepancy")
            else:
                possible_conditions.append("Mild Hip Misalignment: May indicate Muscle Imbalance")

        # Check knee alignment
        knee_alignment = abs(left_knee_px[1] - right_knee_px[1])
        if knee_alignment > 20:  # Threshold for misalignment in pixels
            posture_corrections.append("Knee Misalignment")
            possible_conditions.append("Knee Misalignment: May indicate Joint Issues or Gait Problems")

    # Check spinal angle for overall posture
    # if abs(spine_angle) > 5:  # Threshold for spinal alignment issues
    #     posture_corrections.append("Spinal Misalignment")
    #     if abs(spine_angle) > 15:
    #         possible_conditions.append("Severe Spinal Misalignment: Consider Scoliosis")
    #     else:
    #         possible_conditions.append("Mild Spinal Misalignment: May indicate Poor Posture")

    # Check overall posture
    if len(posture_corrections) == 0:
        return "Good Posture", "No obvious conditions detected"
    else:
        return ", ".join(posture_corrections), "; ".join(possible_conditions)

# Start capturing video (or you can use a specific image)
cap = cv2.VideoCapture(0)

# Set the window to full screen
cv2.namedWindow('Activity and Posture Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Activity and Posture Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detect activity and check posture
        activity, posture_conditions = detect_activity_and_posture(results.pose_landmarks.landmark, frame_width, frame_height)

        # Display the detected activity and posture conditions on the frame
        cv2.putText(frame, f"Activity: {activity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Posture: {posture_conditions}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Activity and Posture Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
