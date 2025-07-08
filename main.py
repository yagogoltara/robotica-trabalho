import cv2
import threading
import queue
import sounddevice as sd
import speech_recognition as sr
import numpy as np
import noisereduce as nr

def audio_producer(q: queue.Queue, stop_event: threading.Event, duration: float, fs: int):
    """Grava blocos de áudio, aplica redução de ruído e os coloca na fila."""
    print("[Áudio] Produtor iniciado. Fale para transcrição.")
    while not stop_event.is_set():
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        audio_float = audio.astype(np.float32) / 32767.0
        
        reduced_noise_audio = nr.reduce_noise(
            y=audio_float.flatten(), 
            sr=fs,
            prop_decrease=0.8 # Fator de redução de ruído (ajustável)
        )
        
        audio_clean = (reduced_noise_audio * 32767.0).astype(np.int16)
        q.put(audio_clean.copy())

def audio_consumer(q: queue.Queue, stop_event: threading.Event, fs: int):
    """Consome áudio da fila e realiza a transcrição via API do Google."""
    recognizer = sr.Recognizer()
    sample_width = 2
    while not (stop_event.is_set() and q.empty()):
        try:
            audio = q.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_data = sr.AudioData(audio.tobytes(), fs, sample_width)

        try:
            texto = recognizer.recognize_google(audio_data, language='pt-BR')
            print(f"[Transcrição] {texto}")
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"[Transcrição] Erro na API; {e}")

        q.task_done()

def main():
    # --- Parâmetros da Aplicação ---
    AUDIO_DURATION = 5.0
    SAMPLE_RATE = 44100
    RECT_COLOR, TEXT_COLOR = (255, 0, 0), (0, 255, 0)
    FONT, FONT_SCALE, FONT_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    WINDOW_NAME = "Webcam (c=cap, q=sair)"

    # --- Configuração das Threads ---
    audio_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    producer_thread = threading.Thread(target=audio_producer, args=(audio_queue, stop_event, AUDIO_DURATION, SAMPLE_RATE), daemon=True)
    consumer_thread = threading.Thread(target=audio_consumer, args=(audio_queue, stop_event, SAMPLE_RATE), daemon=True)
    
    producer_thread.start()
    consumer_thread.start()

    # --- Configuração do OpenCV ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Não foi possível acessar a webcam")
        return

    img_counter = 0
    print("Pressione 'c' para capturar um rosto ou 'q' para sair.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), RECT_COLOR, 2)
                cv2.putText(frame, f"X: {x}, Y: {y}", (x, y-10), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    name = f"face_{img_counter}.png"
                    cv2.imwrite(name, face_img)
                    print(f"📸 Rosto capturado e salvo como: {name}")
                    img_counter += 1
            elif key == ord('q'):
                break
    finally:
        print("Encerrando...")
        stop_event.set()
        producer_thread.join(timeout=1)
        consumer_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()