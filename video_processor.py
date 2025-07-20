import cv2
from typing import Optional, Tuple
import numpy as np
class VideoProcessor:
    def __init__(self, camera_index=0, window_name="Webcam"):
        """
        Initialize the VideoProcessor with camera settings.
        
        Args:
            camera_index (int): Index of the camera to use
            window_name (str): Name of the display window
        """
        self.camera_index = camera_index
        self.window_name = window_name
        self.cap = None
        self.face_detector = None
    
    def initialize_camera(self) -> bool:
        """
        Initialize the camera capture.
        
        Returns:
            bool: True if camera was initialized successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()
    
    def set_face_detector(self, face_detector):
        """
        Set the face detector to use for face detection.
        
        Args:
            face_detector: Instance of FaceDetector
        """
        self.face_detector = face_detector
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the next frame from the camera.
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the captured image
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame (detect faces and draw them).
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame with face detection
        """
        if self.face_detector:
            faces = self.face_detector.detect_faces(frame)
            if len(faces) > 0:
                frame = self.face_detector.draw_face_rectangles(frame, faces)
                frame = self.face_detector.draw_face_coordinates(frame, faces)
        return frame
    
    def show_frame(self, frame: np.ndarray):
        """
        Display a frame in the window.
        
        Args:
            frame: Frame to display
        """
        cv2.imshow(self.window_name, frame)
    
    def capture_face(self, frame: np.ndarray, output_path: str) -> bool:
        """
        Capture and save the first detected face in the frame.
        
        Args:
            frame: Input frame containing faces
            output_path: Path to save the captured face
            
        Returns:
            bool: True if face was captured and saved, False otherwise
        """
        if not self.face_detector:
            return False
            
        faces = self.face_detector.detect_faces(frame)
        if len(faces) == 0:
            return False
            
        face_img = self.face_detector.extract_face(frame, faces[0])
        if face_img is not None:
            cv2.imwrite(output_path, face_img)
            return True
        return False
    
    def release(self):
        """Release camera resources and close windows."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
