# Documentação do Projeto de Processamento de Áudio e Vídeo

Este documento descreve a estrutura e funcionalidades do sistema de processamento de áudio e vídeo para detecção de rostos e transcrição de fala.

## Estrutura do Projeto

```
.
├── application.py      # Classe principal da aplicação
├── audio_processor.py # Processamento de áudio e transcrição
├── face_detector.py   # Detecção e manipulação de rostos
├── main.py            # Script principal (legado)
├── video_processor.py # Processamento de vídeo e captura de frames
├── requirements.txt   # Dependências do projeto
└── setup.sh           # Script de configuração
```

## Módulos

### application.py

Classe principal que orquestra a aplicação, integrando os módulos de áudio e vídeo.

**Classe Principal**: `Application`
- `__init__(camera_index=0, sample_rate=44100, audio_duration=3.0)`: Inicializa a aplicação com os parâmetros fornecidos.
- `run()`: Inicia o loop principal da aplicação.
- `cleanup()`: Libera recursos e encerra a aplicação.

**Fluxo de Execução**:
1. Inicializa processadores de áudio e vídeo
2. Configura detector de rostos
3. Inicia threads de processamento de áudio
4. Executa loop principal de captura de vídeo
5. Gerencia eventos de teclado para captura de rostos

### audio_processor.py

Responsável pelo processamento de áudio e transcrição de fala.

**Classe Principal**: `AudioProcessor`
- `__init__(sample_rate=44100, audio_duration=3.0, queue_size=5)`: Configura o processador de áudio.
- `record_audio()`: Grava áudio com redução de ruído e coloca na fila.
- `transcribe_audio()`: Transcreve áudio da fila usando reconhecimento de fala.
- `start_processing()`: Inicia as threads de gravação e transcrição.
- `stop_processing()`: Para o processamento e limpa recursos.

**Funcionalidades**:
- Gravação de áudio em tempo real
- Redução de ruído
- Transcrição de fala usando Google Speech Recognition
- Processamento assíncrono com fila de áudio

### face_detector.py

Implementa a detecção e manipulação de rostos em imagens.

**Classe Principal**: `FaceDetector`
- `__init__(scale_factor=1.1, min_neighbors=5, min_size=(30, 30))`: Configura o detector de rostos.
- `detect_faces(frame)`: Detecta rostos em um frame de vídeo.
- `draw_face_rectangles(frame, faces, color, thickness)`: Desenha retângulos ao redor dos rostos.
- `draw_face_coordinates(frame, faces, color, font_scale, thickness)`: Adiciona coordenadas dos rostos.
- `extract_face(frame, face_rect)`: Extrai a região de um rosto da imagem.

**Características**:
- Baseado no classificador Haar Cascade
- Suporte a múltiplos rostos
- Métodos auxiliares para visualização

### video_processor.py

Gerencia a captura e processamento de vídeo da webcam.

**Classe Principal**: `VideoProcessor`
- `__init__(camera_index=0, window_name="Webcam")`: Inicializa o processador de vídeo.
- `initialize_camera()`: Inicializa a câmera.
- `set_face_detector(face_detector)`: Configura o detector de rostos.
- `get_frame()`: Captura um frame da câmera.
- `process_frame(frame)`: Processa um frame (detecta rostos).
- `show_frame(frame)`: Exibe um frame na janela.
- `capture_face(frame, output_path)`: Salva um rosto detectado.
- `release()`: Libera os recursos da câmera.

**Funcionalidades**:
- Captura de vídeo em tempo real
- Integração com detector de rostos
- Exibição de frames processados
- Captura e salvamento de rostos

### main.py (Legado)

Script original que implementa a funcionalidade básica de forma procedural.

**Funções Principais**:
- `audio_producer()`: Grava áudio e aplica redução de ruído.
- `audio_consumer()`: Transcreve áudio da fila.
- `main()`: Função principal que gerencia o fluxo do programa.

## Requisitos

- Python 3.7+
- OpenCV
- NumPy
- SoundDevice
- SpeechRecognition
- Noisereduce

## Instalação

1. Clone o repositório
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script principal:
   ```bash
   python application.py
   ```

## Uso

1. A aplicação iniciará a câmera e o microfone
2. Rostos detectados serão destacados na tela
3. Pressione 'c' para capturar o rosto detectado
4. Pressione 'q' para sair da aplicação

## Observações

- A transcrição de áudio requer conexão com a internet
- A qualidade da detecção pode variar com a iluminação e ângulo da câmera
- Arquivos de captura são salvos no diretório 'captures/'

- Ajuste `prop_decrease` em `nr.reduce_noise()` para calibrar a intensidade da redução de ruído.  
- A divisão em threads garante que a UI da webcam não seja bloqueada pela gravação/transcrição de áudio.  
- A comunicação via `queue.Queue` torna o pipeline robusto a picos de carga, evitando perda de dados.  
