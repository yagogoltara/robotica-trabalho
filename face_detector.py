import cv2
import numpy as np
class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Initialize the FaceDetector with detection parameters.
        
        Args:
            scale_factor (float): Scale factor for image pyramid
            min_neighbors (int): Minimum neighbors for detection
            min_size (tuple): Minimum face size (width, height)
        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
    
    def detect_faces(self, frame):
        """
        Detect faces in a given frame.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            list: List of face rectangles (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        return faces
    
    @staticmethod
    def draw_face_rectangles(frame, faces, color=(255, 0, 0), thickness=2):
        """
        Draw rectangles around detected faces.
        
        Args:
            frame: Input image
            faces: List of face rectangles (x, y, w, h)
            color: Rectangle color (B, G, R)
            thickness: Rectangle thickness
            
        Returns:
            Frame with drawn rectangles
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        return frame
    
    @staticmethod
    def draw_face_coordinates(frame, faces, color=(0, 255, 0), font_scale=0.6, thickness=2):
        """
        Draw coordinates of detected faces.
        
        Args:
            frame: Input image
            faces: List of face rectangles (x, y, w, h)
            color: Text color (B, G, R)
            font_scale: Font scale
            thickness: Text thickness
            
        Returns:
            Frame with drawn coordinates
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for (x, y, w, h) in faces:
            cv2.putText(
                frame, 
                f"X: {x}, Y: {y}", 
                (x, y-10), 
                font, 
                font_scale, 
                color, 
                thickness
            )
        return frame
    
    @staticmethod
    def extract_face(frame, face_rect):
        """
        Extract face region from frame.
        
        Args:
            frame: Input image
            face_rect: Face rectangle (x, y, w, h)
            
        Returns:
            Extracted face image or None if invalid rectangle
        """
        x, y, w, h = face_rect
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            return frame[y:y+h, x:x+w]
        return None
