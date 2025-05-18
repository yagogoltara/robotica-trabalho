import cv2

def main():
    # Carrega o classificador pré-treinado para detecção de faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Abre a webcam (0 = câmera padrão)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível acessar a webcam")
        return

    img_counter = 0

    print("Pressione 'c' para capturar o rosto, 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame da webcam")
            break

        # Converte para escala de cinza (necessário para o detector Haar)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta faces no frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Desenha retângulos em volta de cada face detectada
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Mostra o vídeo ao vivo
        cv2.imshow("Webcam (c = captura, q = sair)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Se houver ao menos uma face, salva a primeira detectada
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
