import cv2
import os
from typing import Optional
from audio_processor import AudioProcessor
from video_processor import VideoProcessor
from face_detector import FaceDetector
import numpy as np

class Application:
    def __init__(self, camera_index=0, sample_rate=44100, audio_duration=3.0):
        """
        Inicializa a aplicação com os componentes de áudio e vídeo.
        
        Args:
            camera_index (int): Índice da câmera que vai usar
            sample_rate (int): Taxa de amostragem do áudio em Hz
            audio_duration (float): Duração de cada trecho de áudio em segundos
        """
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            audio_duration=audio_duration
        )
        
        self.face_detector = FaceDetector()
        
        self.video_processor = VideoProcessor(
            camera_index=camera_index,
            window_name="Webcam (c=cap, q=sair)"
        )
        self.video_processor.set_face_detector(self.face_detector)
        
        self.img_counter = 0
        self.output_dir = "captures"
        self._create_output_dir()
    
    def _create_output_dir(self):
        """Cria a pasta pra salvar os rostos capturados, se não existir ainda."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def run(self):
        """Roda o loop principal da aplicação."""
        if not self.video_processor.initialize_camera():
            print("❌ Não foi possível acessar a webcam")
            return
        
        print("Pressione 'c' para capturar um rosto ou 'q' para sair.")
        self.audio_processor.start_processing()
        
        try:
            while True:
                # Pega e processa o frame
                ret, frame = self.video_processor.get_frame()
                if not ret:
                    break
                
                # Processa o frame (detecta e desenha os rostos)
                processed_frame = self.video_processor.process_frame(frame)
                self.video_processor.show_frame(processed_frame)
                
                # Lida com o teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    output_path = os.path.join(
                        self.output_dir, 
                        f"face_{self.img_counter}.png"
                    )
                    if self.video_processor.capture_face(frame, output_path):
                        print(f"📸 Rosto capturado e salvo como: {output_path}")
                        self.img_counter += 1
                    else:
                        print("⚠ Nenhum rosto detectado para captura")
                elif key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nAplicação interrompida pelo usuário")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Libera os recursos usados."""
        self.audio_processor.stop_processing()
        self.video_processor.release()
        print("Aplicação encerrada com sucesso.")


def main():
    """Ponto de entrada principal da aplicação."""
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
