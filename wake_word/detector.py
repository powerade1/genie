import pyaudio
import numpy as np
from openwakeword.model import Model

from config import WAKE_WORD_FRAME_SIZE, WAKE_WORD_MODEL_PATH, WAKE_WORD_THRESHOLD


class WakeWordDetector:
    def __init__(
        self,
        model_path: str = WAKE_WORD_MODEL_PATH,
        threshold: float = WAKE_WORD_THRESHOLD,
    ):
        self.model = Model(wakeword_models=[model_path])
        self.threshold = threshold
        self.audio = pyaudio.PyAudio()
        self._stream = None

    def listen(self, callback):
        """Wake word 감지 시 callback 실행"""
        self._stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=WAKE_WORD_FRAME_SIZE,
        )
        try:
            while True:
                audio_chunk = self._stream.read(WAKE_WORD_FRAME_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                predictions = self.model.predict(audio_data)

                for _, score in predictions.items():
                    if score > self.threshold:
                        print(f"Wake word 감지! (score: {score:.2f})")
                        callback()
                        self.model.reset()
                        break
        finally:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None

    def close(self):
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self.audio.terminate()
