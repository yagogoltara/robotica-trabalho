import cv2
import threading
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import tempfile
import os

def audio_loop(stop_event, duration=5, fs=44100):
    """
    Loop que grava blocos de áudio de `duration` segundos e
    os transcreve, até que stop_event seja setado.
    """
    recognizer = sr.Recognizer()
    while not stop_event.is_set():
        print("🔴 Gravando áudio...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Salva em arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        sf.write(wav_path, audio, fs)

        # Transcreve
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        try:
            texto = recognizer.recognize_google(audio_data, language='pt-BR')
            print(f"[Transcrição] {texto}")
        except sr.UnknownValueError:
            print("[Transcrição] não foi possível entender o áudio")
        except sr.RequestError as e:
            print(f"[Transcrição] erro na API de Speech-to-Text; {e}")

        # Remove arquivo temporário
        os.remove(wav_path)

def main():
    # Carrega Haar Cascade para detecção de faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Abre a webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Não foi possível acessar a webcam")
        return

    # Prepara thread de áudio
    stop_event = threading.Event()
    thread_audio = threading.Thread(target=audio_loop, args=(stop_event, 5))
    thread_audio.daemon = True
    thread_audio.start()

    img_counter = 0
    print("Pressione 'c' para capturar o rosto, 'q' para sair.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Falha ao capturar frame da webcam")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Desenha retângulos nas faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Webcam (c = captura, q = sair)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    img_name = f"face_{img_counter}.png"
                    cv2.imwrite(img_name, face_img)
                    print(f"[+] {img_name} salvo!")
                    img_counter += 1
                else:
                    print("[-] Nenhuma face detectada para capturar.")
            elif key == ord('q'):
                print("Saindo...")
                break
    finally:
        # Para a thread de áudio e aguarda término
        stop_event.set()
        thread_audio.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
