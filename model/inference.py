import torch
import torch.nn as nn
import numpy as np
import threading
import time

class AsyncInference:
    def __init__(self, model: nn.Module):
        self.input_data = None
        self.output_data = None
        self.model = model
        self.device = next(self.model.parameters()).device

        self._event = threading.Event()
        self._lock = threading.Lock()
        self.infer_thread = threading.Thread(target=self._inference, daemon=True)
        self.infer_thread.start()

    def _inference(self):
        while not self._event.is_set():
            with self._lock:
                data = self.input_data
                self.input_data = None   # consume once
            if data is None:
                time.sleep(0.005)        # yield CPU, avoid busy spin
                continue

            img, turn_signal = data
            inp = torch.from_numpy(np.ascontiguousarray(img)).float()
            inp = inp.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True) / 255.0

            with torch.no_grad():
                output = self.model(inp, turn_signal).detach().cpu().numpy()[0]

            with self._lock:
                self.output_data = output

    def put(self, img: np.ndarray, turn_signal: int):
        with self._lock:
            self.input_data = (img, turn_signal)

    def get(self, fallback=None):
        with self._lock:
            return self.output_data if self.output_data is not None else fallback

    def stop(self):
        self._event.set()
        self.infer_thread.join()