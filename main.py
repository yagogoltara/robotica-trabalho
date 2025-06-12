import cv2
import threading
import queue
import sounddevice as sd
import speech_recognition as sr

def audio_producer(q: queue.Queue, stop_event: threading.Event, duration: float, fs: int):
    """
    Grava blocos de áudio de `duration` segundos e coloca a raw data na fila.
    """
    while not stop_event.is_set():
        # gravação de forma não-bloqueante (apenas espera no final)
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        q.put(audio.copy())

def audio_consumer(q: queue.Queue, stop_event: threading.Event, fs: int):
    """
    Consome blocos de áudio da fila e faz transcrição em background.
    """
    recognizer = sr.Recognizer()
    sample_width = 2  # bytes, pois dtype='int16'
    while not (stop_event.is_set() and q.empty()):
        try:
            audio = q.get(timeout=0.5)
        except queue.Empty:
            continue

        # Converte o bloco numpy para sr.AudioData
        audio_bytes = audio.tobytes()
        audio_data = sr.AudioData(audio_bytes, fs, sample_width)

        try:
            texto = recognizer.recognize_google(audio_data, language='pt-BR')
            print(f"[Transcrição] {texto}")
        except sr.UnknownValueError:
            print("[Transcrição] não entendi")
        except sr.RequestError as e:
            print(f"[Transcrição] erro na API; {e}")

        q.task_done()

def main():
    # parâmetros de áudio
    DURATION = 5.0   # segundos por bloco
    FS = 44100       # sample rate

    # fila e evento de parada
    audio_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # threads de áudio
    prod = threading.Thread(target=audio_producer, args=(audio_queue, stop_event, DURATION, FS), daemon=True)
    cons = threading.Thread(target=audio_consumer, args=(audio_queue, stop_event, FS), daemon=True)
    prod.start()
    cons.start()

    # setup webcam e detecção de faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Não foi possível acessar a webcam")
        return

    img_counter = 0
    print("Pressione 'c' para capturar rosto, 'q' para sair.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erro ao capturar frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            cv2.imshow("Webcam (c=cap, q=sair)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if faces.any():
                    x,y,w,h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    name = f"face_{img_counter}.png"
                    cv2.imwrite(name, face_img)
                    print(f"[+] {name} salvo")
                    img_counter += 1
                else:
                    print("[-] Nenhuma face detectada")
            elif key == ord('q'):
                break
    finally:
        # sinaliza parada e aguarda filas esvaziarem
        stop_event.set()
        prod.join(timeout=1)
        cons.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
