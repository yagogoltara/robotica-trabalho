import cv2
import threading
import queue
import sounddevice as sd
import speech_recognition as sr
import numpy as np
import noisereduce as nr # Importa a biblioteca

# --- Funções de Áudio (com alteração no producer) ---

def audio_producer(q: queue.Queue, stop_event: threading.Event, duration: float, fs: int):
    """
    Grava blocos de áudio, aplica redução de ruído e coloca os dados limpos na fila.
    """
    print("[Áudio] Produtor iniciado. Fale para transcrição.")
    while not stop_event.is_set():
        # Gravação de forma não-bloqueante
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # --- PONTO CHAVE: REDUÇÃO DE RUÍDO ---
        # A biblioteca espera um array numpy float, então convertemos e normalizamos
        audio_float = audio.astype(np.float32) / 32767.0
        
        # Aplica a redução de ruído. A biblioteca identifica o ruído automaticamente.
        reduced_noise_audio = nr.reduce_noise(
            y=audio_float.flatten(), # Usa o áudio gravado
            sr=fs,
            prop_decrease=0.8 # Reduz o ruído em 80% (ajustável)
        )
        
        # Converte de volta para o formato original int16
        audio_clean = (reduced_noise_audio * 32767.0).astype(np.int16)
        
        # Coloca o áudio LIMPO na fila
        q.put(audio_clean.copy())


def audio_consumer(q: queue.Queue, stop_event: threading.Event, fs: int):
    """Consome blocos de áudio da fila e faz transcrição (sem alterações)."""
    recognizer = sr.Recognizer()
    sample_width = 2
    while not (stop_event.is_set() and q.empty()):
        try:
            audio = q.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_bytes = audio.tobytes()
        audio_data = sr.AudioData(audio_bytes, fs, sample_width)

        try:
            texto = recognizer.recognize_google(audio_data, language='pt-BR')
            print(f"[Transcrição] {texto}")
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"[Transcrição] erro na API; {e}")

        q.task_done()

# --- Função Principal (sem alterações) ---
def main():
    DURATION = 5.0
    FS = 44100
    RECT_COLOR, TEXT_COLOR = (255, 0, 0), (0, 255, 0)
    FONT, FONT_SCALE, FONT_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    WINDOW_NAME = "Webcam (c=cap, q=sair)"

    audio_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    prod = threading.Thread(target=audio_producer, args=(audio_queue, stop_event, DURATION, FS), daemon=True)
    cons = threading.Thread(target=audio_consumer, args=(audio_queue, stop_event, FS), daemon=True)
    prod.start()
    cons.start()

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
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), RECT_COLOR, 2)
                cv2.putText(frame, f"X: {x}, Y: {y}", (x, y-10), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if len(faces) > 0:
                    x,y,w,h = faces[0]
                    name = f"face_{img_counter}.png"
                    cv2.imwrite(name, frame[y:y+h, x:x+w])
                    print(f"📸 Rosto capturado e salvo como: {name}")
                    img_counter += 1
            elif key == ord('q'):
                break
    finally:
        print("Encerrando...")
        stop_event.set()
        prod.join(timeout=1)
        cons.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()