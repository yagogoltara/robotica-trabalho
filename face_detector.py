import cv2
import numpy as np
class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Inicializa o FaceDetector com os parâmetros de detecção.
        
        Args:
            scale_factor (float): Fator de escala pra pirâmide de imagem
            min_neighbors (int): Mínimo de vizinhos pra considerar um rosto
            min_size (tuple): Tamanho mínimo do rosto (largura, altura)
        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
    
    def detect_faces(self, frame):
        """
        Detecta os rostos no frame que chegar.
        
        Args:
            frame: Imagem em BGR
            
        Returns:
            list: Lista de retângulos dos rostos (x, y, w, h)
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
        Desenha os retângulos em volta dos rostos.
        
        Args:
            frame: Imagem
            faces: Lista dos rostos (x, y, w, h)
            color: Cor do retângulo (B, G, R)
            thickness: Espessura da linha
            
        Returns:
            Frame com os retângulos desenhados
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        return frame
    
    @staticmethod
    def draw_face_coordinates(frame, faces, color=(0, 255, 0), font_scale=0.6, thickness=2):
        """
        Escreve as coordenadas dos rostos na imagem.
        
        Args:
            frame: Imagem
            faces: Lista dos rostos (x, y, w, h)
            color: Cor do texto (B, G, R)
            font_scale: Tamanho da fonte
            thickness: Espessura do texto
            
        Returns:
            Frame com as coordenadas desenhadas
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
        Recorta o rosto do frame.
        
        Args:
            frame: Imagem
            face_rect: Retângulo do rosto (x, y, w, h)
            
        Returns:
            Imagem do rosto ou None se o retângulo for inválido
        """
        x, y, w, h = face_rect
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            return frame[y:y+h, x:x+w]
        return None
