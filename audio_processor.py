import queue
import threading
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import noisereduce as nr

class AudioProcessor:
    def __init__(self, sample_rate=44100, audio_duration=3.0, queue_size=5):
        """
        Initialize the AudioProcessor with audio settings.
        
        Args:
            sample_rate (int): Audio sample rate in Hz
            audio_duration (float): Duration of each audio chunk in seconds
            queue_size (int): Maximum size of the audio processing queue
        """
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.recognizer = sr.Recognizer()
        self.sample_width = 2  # 16-bit audio
        
    def record_audio(self):
        """Record audio with noise reduction and put it in the queue."""
        print("[Áudio] Iniciando gravação. Fale para transcrição.")
        while not self.stop_event.is_set():
            # Record audio
            audio = sd.rec(
                int(self.audio_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            sd.wait()
            
            # Apply noise reduction
            audio_float = audio.astype(np.float32) / 32767.0
            reduced_noise_audio = nr.reduce_noise(
                y=audio_float.flatten(),
                sr=self.sample_rate,
                prop_decrease=0.8
            )
            audio_clean = (reduced_noise_audio * 32767.0).astype(np.int16)
            
            # Put in queue if not full
            try:
                self.queue.put(audio_clean.copy(), timeout=0.5)
            except queue.Full:
                print("[Áudio] Fila cheia, descartando amostra de áudio")
    
    def transcribe_audio(self):
        """Transcribe audio from the queue using Google's speech recognition."""
        while not (self.stop_event.is_set() and self.queue.empty()):
            try:
                audio = self.queue.get(timeout=0.5)
                audio_data = sr.AudioData(
                    audio.tobytes(),
                    self.sample_rate,
                    self.sample_width
                )
                
                try:
                    text = self.recognizer.recognize_google(audio_data, language='pt-BR')
                    print(f"[Transcrição] {text}")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"[Transcrição] Erro na API: {e}")
                
                self.queue.task_done()
            except queue.Empty:
                continue
    
    def start_processing(self):
        """Start audio processing threads."""
        self.recording_thread = threading.Thread(
            target=self.record_audio,
            daemon=True
        )
        self.transcription_thread = threading.Thread(
            target=self.transcribe_audio,
            daemon=True
        )
        
        self.recording_thread.start()
        self.transcription_thread.start()
    
    def stop_processing(self):
        """Stop audio processing and clean up resources."""
        self.stop_event.set()
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1)
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join(timeout=1)
        print("[Áudio] Processamento de áudio encerrado")
