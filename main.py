import cv2
import threading
import queue
import sounddevice as sd
import speech_recognition as sr

# --- Funções de Áudio ---

def audio_producer(q: queue.Queue, stop_event: threading.Event, duration: float, fs: int):
    """
    Grava blocos de áudio de `duration` segundos e coloca a raw data na fila.
    """
    while not stop_event.is_set():
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

        audio_bytes = audio.tobytes()
        audio_data = sr.AudioData(audio_bytes, fs, sample_width)

        try:
            texto = recognizer.recognize_google(audio_data, language='pt-BR')
            print(f"[Transcrição] {texto}")
        except sr.UnknownValueError:
            # Silenciando a saída para não poluir o console com "não entendi"
            pass
        except sr.RequestError as e:
            print(f"[Transcrição] erro na API; {e}")

        q.task_done()

# --- Função Principal de Visão Computacional ---

def main():
    # --- Parâmetros de Áudio ---
    DURATION = 5.0
    FS = 44100

    # --- Parâmetros de Vídeo e Detecção ---
    RECT_COLOR = (255, 0, 0)      # Cor do retângulo (BGR -> Azul)
    TEXT_COLOR = (0, 255, 0)      # Cor do texto (BGR -> Verde)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    WINDOW_NAME = "Webcam (c=cap, q=sair)"

    # --- Fila e Evento de Parada para Threads ---
    audio_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # --- Inicialização das Threads de Áudio ---
    prod = threading.Thread(target=audio_producer, args=(audio_queue, stop_event, DURATION, FS), daemon=True)
    cons = threading.Thread(target=audio_consumer, args=(audio_queue, stop_event, FS), daemon=True)
    prod.start()
    cons.start()

    # --- Setup da Webcam e Classificador de Rosto ---
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
                print("❌ Erro ao capturar frame")
                break

            # A detecção de faces é mais eficiente em escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Itera sobre cada rosto detectado
            for (x, y, w, h) in faces:
                # 1. Desenha o retângulo ao redor do rosto
                cv2.rectangle(frame, (x, y), (x + w, y + h), RECT_COLOR, 2)

                # 2. Prepara e exibe as coordenadas X e Y
                coord_text = f"X: {x}, Y: {y}"
                # Posiciona o texto um pouco acima do retângulo para melhor visualização
                text_position = (x, y - 10)

                cv2.putText(
                    img=frame,
                    text=coord_text,
                    org=text_position,
                    fontFace=FONT,
                    fontScale=FONT_SCALE,
                    color=TEXT_COLOR,
                    thickness=FONT_THICKNESS
                )

            # Mostra o frame final com as anotações
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if len(faces) > 0:
                    # Salva a primeira face detectada
                    x, y, w, h = faces[0]
                    face_img = frame[y:y + h, x:x + w]
                    name = f"face_{img_counter}.png"
                    cv2.imwrite(name, face_img)
                    print(f"📸 Rosto capturado e salvo como: {name}")
                    img_counter += 1
                else:
                    print("[-] Nenhuma face detectada para capturar.")
            elif key == ord('q'):
                print("Encerrando...")
                break
    finally:
        # Garante o encerramento limpo dos recursos
        stop_event.set()
        prod.join(timeout=1)
        cons.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()