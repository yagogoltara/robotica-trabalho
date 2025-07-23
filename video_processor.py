import cv2
from typing import Optional, Tuple
import numpy as np
class VideoProcessor:
    def __init__(self, camera_index=0, window_name="Webcam"):
        """
        Inicializa o VideoProcessor com as configs da câmera.
        
        Args:
            camera_index (int): Índice da câmera que vai usar
            window_name (str): Nome da janela que vai aparecer
        """
        self.camera_index = camera_index
        self.window_name = window_name
        self.cap = None
        self.face_detector = None
    
    def initialize_camera(self) -> bool:
        """
        Inicializa a captura da câmera.
        
        Returns:
            bool: True se conseguiu abrir a câmera, False se deu ruim
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()
    
    def set_face_detector(self, face_detector):
        """
        Define qual detector de rosto vai usar.
        
        Args:
            face_detector: Instância do FaceDetector
        """
        self.face_detector = face_detector
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Pega o próximo frame da câmera.
        
        Returns:
            tuple: (sucesso, frame) onde sucesso é bool e frame é a imagem
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processa um frame (detecta e desenha os rostos).
        
        Args:
            frame: Frame que vai ser processado
            
        Returns:
            Frame processado com os rostos desenhados
        """
        if self.face_detector:
            faces = self.face_detector.detect_faces(frame)
            if len(faces) > 0:
                frame = self.face_detector.draw_face_rectangles(frame, faces)
                frame = self.face_detector.draw_face_coordinates(frame, faces)
        return frame
    
    def show_frame(self, frame: np.ndarray):
        """
        Mostra o frame na janela.
        
        Args:
            frame: Frame que vai aparecer
        """
        cv2.imshow(self.window_name, frame)
    
    def capture_face(self, frame: np.ndarray, output_path: str) -> bool:
        """
        Salva o primeiro rosto detectado no frame.
        
        Args:
            frame: Frame que tem os rostos
            output_path: Caminho pra salvar o rosto
            
        Returns:
            bool: True se salvou o rosto, False se não achou nada
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
        """Libera a câmera e fecha as janelas."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
