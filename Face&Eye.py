import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize video capture
cap = cv2.VideoCapture(0)

# Continuous loop to process webcam frames
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for better landmark visualization
        frame = cv2.flip(frame, 1)

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find face landmarks
        results = face_mesh.process(rgb_frame)

        # Draw face landmarks on the frames
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                ih, iw, _ = frame.shape

                # Define the left eye landmark indices
                LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144, 145, 163, 7, 246, 161, 160, 159]
                LEFT_IRIS_LANDMARKS = [468, 469, 470, 471]

                # Define the right eye landmark indices
                RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380, 374, 390, 249, 466, 388, 387, 386]
                RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476]

                # Draw circles around the left eye landmarks (for visualization)
                for idx in LEFT_EYE_LANDMARKS:
                    x = int(face_landmarks.landmark[idx].x * iw)
                    y = int(face_landmarks.landmark[idx].y * ih)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Draw circles around the right eye landmarks (for visualization)
                for idx in RIGHT_EYE_LANDMARKS:
                    x = int(face_landmarks.landmark[idx].x * iw)
                    y = int(face_landmarks.landmark[idx].y * ih)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Find the bounding box coordinates for the left eye region
                left_eye_bbox = (
                    min([face_landmarks.landmark[idx].x for idx in LEFT_EYE_LANDMARKS]),
                    min([face_landmarks.landmark[idx].y for idx in LEFT_EYE_LANDMARKS]),
                    max([face_landmarks.landmark[idx].x for idx in LEFT_EYE_LANDMARKS]),
                    max([face_landmarks.landmark[idx].y for idx in LEFT_EYE_LANDMARKS])
                )

                # Calculate pixel coordinates for the left eye bounding box with a buffer
                x_min_left = int(left_eye_bbox[0] * iw) - 20
                y_min_left = int(left_eye_bbox[1] * ih) - 20
                x_max_left = int(left_eye_bbox[2] * iw) + 20
                y_max_left = int(left_eye_bbox[3] * ih) + 20

                # Ensure the coordinates are within frame boundaries
                x_min_left = max(x_min_left, 0)
                y_min_left = max(y_min_left, 0)
                x_max_left = min(x_max_left, iw)
                y_max_left = min(y_max_left, ih)

                # Extract the left eye region
                left_eye_region = frame[y_min_left:y_max_left, x_min_left:x_max_left]

                # Calculate pixel coordinates for the right eye bounding box with a buffer
                right_eye_bbox = (
                    min([face_landmarks.landmark[idx].x for idx in RIGHT_EYE_LANDMARKS]),
                    min([face_landmarks.landmark[idx].y for idx in RIGHT_EYE_LANDMARKS]),
                    max([face_landmarks.landmark[idx].x for idx in RIGHT_EYE_LANDMARKS]),
                    max([face_landmarks.landmark[idx].y for idx in RIGHT_EYE_LANDMARKS])
                )

                # Ensure the coordinates are within frame boundaries
                x_min_right = int(right_eye_bbox[0] * iw) - 20
                y_min_right = int(right_eye_bbox[1] * ih) - 20
                x_max_right = int(right_eye_bbox[2] * iw) + 20
                y_max_right = int(right_eye_bbox[3] * ih) + 20

                # Checking if the coordinates are within the frame
                x_min_right = max(x_min_right, 0)
                y_min_right = max(y_min_right, 0)
                x_max_right = min(x_max_right, iw)
                y_max_right = min(y_max_right, ih)

                # Extract the right eye region
                right_eye_region = frame[y_min_right:y_max_right, x_min_right:x_max_right]


                # Find the center of the left iris
                iris_x_left = int(sum([face_landmarks.landmark[idx].x for idx in LEFT_IRIS_LANDMARKS]) / len(LEFT_IRIS_LANDMARKS) * iw)
                iris_y_left = int(sum([face_landmarks.landmark[idx].y for idx in LEFT_IRIS_LANDMARKS]) / len(LEFT_IRIS_LANDMARKS) * ih)


                # Find the center of the right iris
                iris_x_right = int(sum([face_landmarks.landmark[idx].x for idx in RIGHT_IRIS_LANDMARKS]) / len(RIGHT_IRIS_LANDMARKS) * iw)
                iris_y_right = int(sum([face_landmarks.landmark[idx].y for idx in RIGHT_IRIS_LANDMARKS]) / len(RIGHT_IRIS_LANDMARKS) * ih)


                # Calculate relative positions of the iris centers within the eye regions
                iris_x_left_rel = int((iris_x_left - x_min_left) * (300 / (x_max_left - x_min_left)))
                iris_y_left_rel = int((iris_y_left - y_min_left) * (300 / (x_max_left - x_min_left)))
                iris_x_right_rel = int((iris_x_right - x_min_right) * (300 / (x_max_right - x_min_right)))
                iris_y_right_rel = int((iris_y_right - y_min_right) * (300 / (x_max_right - x_min_right)))

                # Display the left eye region
                if left_eye_region.size > 0:
                    aspect_ratio = left_eye_region.shape[1] / left_eye_region.shape[0]
                    new_width = 300
                    new_height = int(new_width / aspect_ratio)
                    left_eye_region_resized = cv2.resize(left_eye_region, (new_width, new_height))
                    cv2.circle(left_eye_region_resized, (iris_x_left_rel, iris_y_left_rel), 3, (0, 0, 255), -1)
                    cv2.imshow('Left Eye Region', left_eye_region_resized)


                # Display the right eye region
                if right_eye_region.size > 0:
                    aspect_ratio = right_eye_region.shape[1] / right_eye_region.shape[0]
                    new_width = 300
                    new_height = int(new_width / aspect_ratio)
                    right_eye_region_resized = cv2.resize(right_eye_region, (new_width, new_height))
                    cv2.circle(right_eye_region_resized, (iris_x_right_rel, iris_y_right_rel), 3, (0, 0, 255), -1)
                    cv2.imshow('Right Eye Region', right_eye_region_resized)


                # Main frame dot
                cv2.circle(frame, (iris_x_left, iris_y_left), 3, (0, 0, 255), -1)
                cv2.circle(frame, (iris_x_right, iris_y_right), 3, (0, 0, 255), -1)

        
        cv2.imshow('Face and Eye Tracking', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
