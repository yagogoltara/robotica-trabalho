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
        Initialize the application with audio and video components.
        
        Args:
            camera_index (int): Index of the camera to use
            sample_rate (int): Audio sample rate in Hz
            audio_duration (float): Duration of each audio chunk in seconds
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
        """Create output directory for captured faces if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def run(self):
        """Run the main application loop."""
        if not self.video_processor.initialize_camera():
            print("❌ Não foi possível acessar a webcam")
            return
        
        print("Pressione 'c' para capturar um rosto ou 'q' para sair.")
        self.audio_processor.start_processing()
        
        try:
            while True:
                # Get and process frame
                ret, frame = self.video_processor.get_frame()
                if not ret:
                    break
                
                # Process frame (detect and draw faces)
                processed_frame = self.video_processor.process_frame(frame)
                self.video_processor.show_frame(processed_frame)
                
                # Handle keyboard input
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
        """Clean up resources."""
        self.audio_processor.stop_processing()
        self.video_processor.release()
        print("Aplicação encerrada com sucesso.")


def main():
    """Main entry point for the application."""
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
