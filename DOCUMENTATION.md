# Documentação do Projeto de Captura de Rosto e Áudio

Este documento descreve as **bibliotecas** utilizadas e a **estrutura** do código, detalhando o propósito de cada componente.

---

## Bibliotecas Utilizadas

- **OpenCV (`cv2`)**
  - Processamento de vídeo e detecção de rostos em tempo real.
  - Utiliza o classificador Haar Cascade (`haarcascade_frontalface_default.xml`) para localizar faces no frame capturado pela webcam.

- **sounddevice**
  - Captura e reprodução de áudio a partir do microfone.
  - Fornece a função `sd.rec()` para gravação de blocos de áudio e `sd.play()` para reprodução.

- **speech_recognition (`sr`)**
  - Conversão de áudio em texto (Speech-to-Text) usando serviços como o Google Web Speech API.
  - Classe `Recognizer` para processar `AudioData` e método `recognize_google()` para obter transcrições.

- **numpy**
  - Manipulação eficiente de arrays numéricos.
  - Conversão entre tipos de dados (`int16` ↔️ `float32`) para processamento de áudio.

- **noisereduce (`nr`)**
  - Redução de ruído de fundo em gravações de áudio.
  - Função `nr.reduce_noise()` aplica algoritmos de supressão de ruído em tempo real.

- **threading**
  - Criação de threads independentes para gravação/processamento de áudio e loop principal da aplicação.
  - Permite rodar captura de áudio sem bloquear a interface de vídeo.

- **queue**
  - Comunicação segura entre threads.
  - Armazena blocos de áudio processados para consumo posterior pela thread responsável pela transcrição.

---

## Estrutura do Código

O código está organizado em **três partes principais**:

1. **Funções de Áudio** (`audio_producer` e `audio_consumer`)  
2. **Função Principal** (`main`)  
3. **Configurações de Parâmetros** (constantes e inicialização)  

### 1. `audio_producer(q, stop_event, duration, fs)`

- **Objetivo**: Gravar blocos de áudio, aplicar redução de ruído e inserir dados limpos em uma fila.  
- **Parâmetros**:
  - `q` (`queue.Queue`): fila para enviar áudio processado.  
  - `stop_event` (`threading.Event`): sinaliza quando parar o loop.  
  - `duration` (`float`): duração de cada bloco de gravação em segundos.  
  - `fs` (`int`): taxa de amostragem (samples por segundo).  

- **Fluxo**:
  1. Chama `sd.rec()` para gravar por `duration` segundos.  
  2. Converte o buffer `int16` para `float32` e normaliza.  
  3. Aplica `nr.reduce_noise()` para remover ruído de fundo (parâmetro `prop_decrease` ajusta intensidade).  
  4. Reconverte o resultado para `int16` e coloca na fila `q`.  

### 2. `audio_consumer(q, stop_event, fs)`

- **Objetivo**: Consumir blocos de áudio da fila e transcrever em texto.  
- **Parâmetros**:
  - `q` (`queue.Queue`): fila contendo blocos de áudio limpos.  
  - `stop_event` (`threading.Event`): sinaliza término da aplicação.  
  - `fs` (`int`): taxa de amostragem usada para criar `AudioData`.  

- **Fluxo**:
  1. Retira dados de áudio da fila (`q.get()`).  
  2. Cria um objeto `sr.AudioData` a partir do buffer de bytes.  
  3. Chama `recognizer.recognize_google()` para obter transcrição em português.  
  4. Imprime o texto reconhecido no console.  

### 3. `main()`

- **Objetivo**: Inicializar a aplicação, threads e loop de vídeo.  

- **Etapas**:
  1. Define constantes de áudio (`DURATION`, `FS`) e cores/fontes para a interface.  
  2. Cria `audio_queue` e `stop_event`.  
  3. Inicia as threads:
     - **Produtor**: captura e processa áudio.  
     - **Consumidor**: transcreve áudio processado.  
  4. Inicializa o detector de faces com OpenCV e abre a webcam.  
  5. Loop principal de vídeo:
     - Captura frames da câmera.  
     - Converte para escala de cinza e detecta faces.  
     - Desenha retângulos e coordenadas (X, Y) sobre cada face.  
     - Exibe janela com o feed ao vivo.  
     - Captura tecla:
       - **`c`**: captura e salva imagem da primeira face detectada.  
       - **`q`**: encerra o loop.  
  6. No bloco `finally`, sinaliza parada (`stop_event.set()`), aguarda término das threads e libera recursos (webcam, janelas).  

---

### Observações

- Ajuste `prop_decrease` em `nr.reduce_noise()` para calibrar a intensidade da redução de ruído.  
- A divisão em threads garante que a UI da webcam não seja bloqueada pela gravação/transcrição de áudio.  
- A comunicação via `queue.Queue` torna o pipeline robusto a picos de carga, evitando perda de dados.  

---

*Fim da documentação.*  
