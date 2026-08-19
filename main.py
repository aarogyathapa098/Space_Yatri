import cv2
import mediapipe as mp
import math
import time
import urllib.request

from pathlib import Path
from collections import deque, Counter


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/hand_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

CAMERA_ID = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Number of recent gesture predictions used for smoothing
SMOOTHING_FRAMES = 7


# ============================================================
# DOWNLOAD MEDIAPIPE MODEL
# ============================================================

def download_model():
    """
    Download the MediaPipe hand landmark model if it does not
    already exist.
    """

    if MODEL_PATH.exists():
        return

    print("Hand landmark model not found.")
    print("Downloading MediaPipe model...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model downloaded successfully.")


# ============================================================
# MATHEMATICAL HELPER FUNCTIONS
# ============================================================

def distance(point1, point2):
    """
    Calculate the 3D Euclidean distance between two MediaPipe
    landmarks.

    Each landmark contains:
        x
        y
        z
    """

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle ABC.

    point_b is the middle point.

    Example:

        A
         \
          B ----- C

    We calculate the angle at B.
    """

    vector_ba = (
        point_a.x - point_b.x,
        point_a.y - point_b.y,
        point_a.z - point_b.z
    )

    vector_bc = (
        point_c.x - point_b.x,
        point_c.y - point_b.y,
        point_c.z - point_b.z
    )

    dot_product = (
        vector_ba[0] * vector_bc[0] +
        vector_ba[1] * vector_bc[1] +
        vector_ba[2] * vector_bc[2]
    )

    magnitude_ba = math.sqrt(
        vector_ba[0] ** 2 +
        vector_ba[1] ** 2 +
        vector_ba[2] ** 2
    )

    magnitude_bc = math.sqrt(
        vector_bc[0] ** 2 +
        vector_bc[1] ** 2 +
        vector_bc[2] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Numerical rounding can occasionally produce values
    # slightly above 1 or below -1.
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle = math.degrees(math.acos(cosine_angle))

    return angle


# ============================================================
# FINGER DETECTION
# ============================================================

def is_finger_extended(
    landmarks,
    mcp_index,
    pip_index,
    tip_index,
    angle_threshold=155
):
    """
    Determine whether a finger is extended.

    We check two things:

    1. Is the finger approximately straight?
    2. Is the fingertip farther from the wrist than the PIP joint?

    This is more flexible than simply checking whether:
        tip.y < pip.y

    because the hand does not always have to be perfectly upright.
    """

    wrist = landmarks[0]

    mcp = landmarks[mcp_index]
    pip = landmarks[pip_index]
    tip = landmarks[tip_index]

    finger_angle = calculate_angle(
        mcp,
        pip,
        tip
    )

    tip_distance = distance(
        tip,
        wrist
    )

    pip_distance = distance(
        pip,
        wrist
    )

    finger_is_straight = finger_angle > angle_threshold

    tip_is_farther = tip_distance > pip_distance * 1.05

    return finger_is_straight and tip_is_farther


def is_thumb_extended(landmarks):
    """
    Detect whether the thumb is extended.

    Thumb landmarks:

        1 = CMC
        2 = MCP
        3 = IP
        4 = TIP

    The thumb behaves differently from the other four fingers,
    so it requires separate logic.
    """

    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    index_mcp = landmarks[5]
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    # Check whether thumb itself is relatively straight
    thumb_angle = calculate_angle(
        thumb_mcp,
        thumb_ip,
        thumb_tip
    )

    # Estimate hand size
    palm_size = distance(
        wrist,
        middle_mcp
    )

    # Check how far the thumb is from the index finger
    thumb_separation = distance(
        thumb_tip,
        index_mcp
    )

    thumb_is_straight = thumb_angle > 150

    thumb_is_away_from_hand = (
        thumb_separation > palm_size * 0.65
    )

    return thumb_is_straight and thumb_is_away_from_hand


# ============================================================
# GESTURE CLASSIFICATION
# ============================================================

def classify_gesture(landmarks):
    """
    Convert the 21 hand landmarks into one gesture.

    Supported gestures:

        OPEN PALM
        FIST
        POINT
        VICTORY
        THUMBS UP
        THUMBS DOWN
        UNKNOWN
    """

    # ---------------------------------------------------------
    # Finger landmark numbers
    # ---------------------------------------------------------

    # Index finger:
    # MCP = 5
    # PIP = 6
    # TIP = 8

    index_extended = is_finger_extended(
        landmarks,
        5,
        6,
        8
    )

    # Middle finger
    middle_extended = is_finger_extended(
        landmarks,
        9,
        10,
        12
    )

    # Ring finger
    ring_extended = is_finger_extended(
        landmarks,
        13,
        14,
        16
    )

    # Pinky
    pinky_extended = is_finger_extended(
        landmarks,
        17,
        18,
        20
    )

    thumb_extended = is_thumb_extended(
        landmarks
    )

    fingers = [
        index_extended,
        middle_extended,
        ring_extended,
        pinky_extended
    ]

    # ---------------------------------------------------------
    # OPEN PALM
    # ---------------------------------------------------------

    if all(fingers):
        return "OPEN PALM"

    # ---------------------------------------------------------
    # VICTORY / PEACE SIGN
    # ---------------------------------------------------------

    if (
        index_extended
        and middle_extended
        and not ring_extended
        and not pinky_extended
    ):
        return "VICTORY"

    # ---------------------------------------------------------
    # POINTING
    # ---------------------------------------------------------

    if (
        index_extended
        and not middle_extended
        and not ring_extended
        and not pinky_extended
    ):
        return "POINT"

    # ---------------------------------------------------------
    # THUMB / FIST GESTURES
    # ---------------------------------------------------------

    if not any(fingers):

        if thumb_extended:

            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]

            wrist = landmarks[0]
            middle_mcp = landmarks[9]

            palm_size = distance(
                wrist,
                middle_mcp
            )

            # OpenCV image coordinates:
            #
            # y = 0
            # ↑ top
            #
            # ↓ bottom
            #
            # Therefore:
            # smaller y = higher position

            vertical_difference = (
                thumb_ip.y - thumb_tip.y
            )

            # Thumb points upward
            if vertical_difference > palm_size * 0.25:
                return "THUMBS UP"

            # Thumb points downward
            if vertical_difference < -palm_size * 0.25:
                return "THUMBS DOWN"

        return "FIST"

    return "UNKNOWN"


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand(frame, landmarks):
    """
    Draw the 21 landmarks and the connections between them.
    """

    height, width, _ = frame.shape

    # MediaPipe already defines which landmarks should connect
    connections = (
        mp.tasks.vision
        .HandLandmarksConnections
        .HAND_CONNECTIONS
    )

    # ---------------------------------------------------------
    # Draw connections
    # ---------------------------------------------------------

    for connection in connections:

        start = landmarks[connection.start]
        end = landmarks[connection.end]

        start_point = (
            int(start.x * width),
            int(start.y * height)
        )

        end_point = (
            int(end.x * width),
            int(end.y * height)
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (255, 255, 255),
            2
        )

    # ---------------------------------------------------------
    # Draw individual landmarks
    # ---------------------------------------------------------

    for index, landmark in enumerate(landmarks):

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

        # Display landmark number
        cv2.putText(
            frame,
            str(index),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1
        )


# ============================================================
# GESTURE SMOOTHING
# ============================================================

gesture_history = deque(
    maxlen=SMOOTHING_FRAMES
)


def smooth_gesture(new_gesture):
    """
    Prevent gesture names from rapidly flickering.

    Instead of trusting one frame, we examine several
    recent predictions and use the most common one.
    """

    gesture_history.append(new_gesture)

    counts = Counter(gesture_history)

    most_common_gesture, count = counts.most_common(1)[0]

    # Require a majority of the stored frames
    required_votes = len(gesture_history) // 2 + 1

    if count >= required_votes:
        return most_common_gesture

    return "UNKNOWN"


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    download_model()

    # ---------------------------------------------------------
    # Configure MediaPipe
    # ---------------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=RunningMode.VIDEO,

        # Only detect one hand for Phase 1
        num_hands=1,

        min_hand_detection_confidence=0.6,

        min_hand_presence_confidence=0.6,

        min_tracking_confidence=0.6
    )

    # ---------------------------------------------------------
    # Open webcam
    # ---------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_ID
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Timestamp required by MediaPipe VIDEO mode
    start_time = time.perf_counter()

    previous_time = time.perf_counter()

    # ---------------------------------------------------------
    # Create MediaPipe detector
    # ---------------------------------------------------------

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read frame.")
                break

            # Mirror image like a selfie camera
            frame = cv2.flip(
                frame,
                1
            )

            # -------------------------------------------------
            # OpenCV uses BGR
            # MediaPipe expects RGB
            # -------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Convert NumPy image to MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # -------------------------------------------------
            # Generate timestamp
            # -------------------------------------------------

            timestamp_ms = int(
                (time.perf_counter() - start_time)
                * 1000
            )

            # -------------------------------------------------
            # Detect hand landmarks
            # -------------------------------------------------

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            detected_gesture = "NO HAND"

            # -------------------------------------------------
            # Process detected hand
            # -------------------------------------------------

            if result.hand_landmarks:

                landmarks = (
                    result.hand_landmarks[0]
                )

                # Draw hand skeleton
                draw_hand(
                    frame,
                    landmarks
                )

                raw_gesture = classify_gesture(
                    landmarks
                )

                detected_gesture = smooth_gesture(
                    raw_gesture
                )

            else:

                # Remove old predictions when hand disappears
                gesture_history.clear()

            # -------------------------------------------------
            # FPS calculation
            # -------------------------------------------------

            current_time = time.perf_counter()

            elapsed = (
                current_time - previous_time
            )

            if elapsed > 0:
                fps = 1 / elapsed
            else:
                fps = 0

            previous_time = current_time

            # -------------------------------------------------
            # Display information
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (0, 0),
                (640, 90),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Gesture: {detected_gesture}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Space Gesture Controller - Phase 1",
                frame
            )

            # Press Q to quit
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()