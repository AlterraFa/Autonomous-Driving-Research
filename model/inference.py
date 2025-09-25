# Helper group
import os, re
import time
import threading
import importlib
import inspect

# Compute group
import cv2
import torch
import numpy as np
from model.tensor_engine import ImageTensorRTInference

# Cuda group
import pycuda.driver as cuda
import pycuda.autoinit

# Logging group
from traceback import print_exc
from utils.messages.logger import Logger
if not hasattr(np, "float"): np.float = np.float64

torch.set_float32_matmul_precision("highest")

class ModelLoader:
    def __init__(self):
        self.log = Logger()
        pass
    
    def _extract_class(self, path: str):
        """
        Extract the class object from a checkpoint path.

        Args:
            path (str): Path to checkpoint file 
                        (e.g. 'model/PilotNet/PilotNetExperiment/run5/best_PilotNetStatic_run5.pt')

        Returns:
            type: Class object found in the project structure
        """
        self.log.DEBUG(f"Automatically finding module for the specified model")
        
        fname = os.path.basename(path)
        match = re.search(r"best_(.+?)_run\d+\.(?:pt|engine)$", fname)
        if not match:
            raise ValueError(f"Could not parse class name from filename: {fname}")
        class_name = match.group(1)

        dir_path = os.path.dirname(path)
        while dir_path and os.path.basename(dir_path) != "model":
            dir_path = os.path.dirname(dir_path)

        if not dir_path:
            self.log.ERROR("Could not find [bold][i]'model.py'[/][/] file in path", exit_code = 12)


        rel_path = os.path.relpath(os.path.dirname(path), dir_path)  
        parts = rel_path.split(os.sep)

        candidate_modules = []
        for i in range(len(parts)):
            subpath = ".".join(parts[:i+1])
            candidate_modules.append(f"model.{subpath}.model")
        candidate_modules.append("model")  # fallback

        for module_path in candidate_modules:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name) and inspect.isclass(getattr(module, class_name)):
                    self.log.INFO(f"Found module with class: [bold]{class_name}[/] in [bold]{module_path}[/] module")
                    return getattr(module, class_name)
            except ModuleNotFoundError:
                continue
        
        self.log.ERROR(f"Could not find class [bold]{class_name}[/] in any of: {candidate_modules}", exit_code = 12)

class AsyncInference:
    def __init__(self):
        self.input_data = None
        self.output_data = None
        
        self._event = threading.Event()
        self._lock = threading.Lock()
        self.infer_thread = threading.Thread(target=self._inference_torch if self.use_tensorrt == False else self._inference_tensorrt, daemon=True)
        self.infer_thread.start()

    def _inference_torch(self):


        while not self._event.is_set():
            with self._lock:
                data = self.input_data
                self.input_data = None   # consume once
            if data is None:
                time.sleep(0.005)        # yield CPU, avoid busy spin
                continue

            inp_img, extra_data = data
            
            if isinstance(inp_img, (torch.Tensor, np.ndarray, cv2.Mat)):
                inp = torch.from_numpy(np.ascontiguousarray(inp_img)).float()
                inp = inp.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True) / 255.0
                inp = [inp]
            else:
                inp = []
                for img in inp_img:
                    inp_tmp = torch.from_numpy(np.ascontiguousarray(img)).float()
                    inp_tmp = inp_tmp.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True) / 255.0
                    inp += [inp_tmp]
                    
            if not isinstance(extra_data, (torch.Tensor, np.ndarray, cv2.Mat)):
                processor_data = extra_data
            else:
                ...

            with torch.no_grad():
                output = self.pytorch(*inp, processor_data).detach().cpu().numpy()[0]
                    
            with self._lock:
                self.output_data = output
        
    def _inference_tensorrt(self):

        self.ctx.push()
        self.log.INFO("Started engine inference")

        while not self._event.is_set():
            with self._lock:
                data = self.input_data
                self.input_data = None   # consume once
            if data is None:
                time.sleep(0.005)        # yield CPU, avoid busy spin
                continue

            inp_img, extra_data = data
            
            if isinstance(inp_img, (torch.Tensor, np.ndarray, cv2.Mat)):
                inp = torch.from_numpy(np.ascontiguousarray(inp_img)).float()
                inp = inp.permute(2, 0, 1).unsqueeze(0) / 255.0
                inp = inp.detach().cpu().numpy().astype(np.float32)
                inp = np.ascontiguousarray(inp)
                inp = [inp]
            else:
                inp = []
                for img in inp_img:
                    inp_tmp = torch.from_numpy(np.ascontiguousarray(img)).float()
                    inp_tmp = inp_tmp.permute(2, 0, 1).unsqueeze(0).to(non_blocking=True) / 255.0
                    inp += [inp_tmp]

            if not isinstance(extra_data, (torch.Tensor, np.ndarray, cv2.Mat)):
                processor_data = extra_data
            else:
                ...
                    
            try:
                raw_output = self.engine.infer(*inp)
                # Unwrap using .__func__ is needed
                output = self.processor.__func__(raw_output, processor_data) # All of the function should have been static method but the first arg is prepend with a self
                
            except Exception as e:
                print_exc()
                break

            with self._lock:
                self.output_data = output
        self.ctx.pop()
        self.log.INFO("Engine inference stopped")

    def put(self, inp: tuple, processor_data: tuple):
        with self._lock:
            self.input_data = (inp, processor_data)

    def get(self, fallback=None):
        with self._lock:
            return self.output_data if self.output_data is not None else fallback

    def stop(self):
        self._event.set()
        self.infer_thread.join()

    @staticmethod
    def _from_checkpoint(cls, path: str, device: str = "cpu", **model_kwargs):
        """
        Load a model checkpoint, automatically finding the class recursively 
        in the package path that contains the checkpoint.

        Args:
            path (str): Path to checkpoint file (e.g. 'model/PilotNet/PilotNetExperiment/run5/best_PilotNetStatic_run5.pt')
            device (str): Device to load model onto ('cpu' or 'cuda')
            **model_kwargs: Extra arguments to pass to the model constructor (needed if state_dict only)

        Returns:
            torch.nn.Module: Loaded model
        """

        # Load checkpoint
        try:
            state_dict = torch.load(path, map_location=device)
            if isinstance(state_dict, dict):
                model = cls(**model_kwargs)
                model.load_state_dict(state_dict)
                model.to(device).eval()
                return model
        except Exception:
            torch.serialization.add_safe_globals([cls])
            model = torch.load(path, weights_only=False, map_location=device)
            model.to(device).eval()
            return model
    
    @classmethod
    def load_model(cls, path: str, device: str = "cpu", **model_kwargs):
        """
        Load a model checkpoint, automatically inferring class name and module path from filename.

        Args:
            path (str): Path to checkpoint file (e.g. 'model/PilotNet/best_PilotNetStatic_run1.pt')
            device (str): Device to load model onto ('cpu' or 'cuda')
            **model_kwargs: Extra arguments to pass to the model constructor (needed if state_dict only)

        Returns:
            torch.nn.Module: Loaded model
        """
        cls.log = Logger()

        fname = os.path.basename(path)
        name, ext = os.path.splitext(fname)
        
        model_class = ModelLoader()._extract_class(path)

        if ext == ".pt":
            cls.log.INFO("Using Dynamic graph model variant (Pytorch)")
            cls.pytorch = cls._from_checkpoint(model_class, path, device = device)
            cls.device = next(cls.pytorch.parameters()).device
            cls.use_tensorrt = False
            return cls()
        elif ext == '.engine':
            cls.log.INFO("Using Engine variant")
            if hasattr(model_class, "postprocessor"):
                # Getattr actually does take a function wrapped in staticmethod but because we invoked cls.processor, the class prepend self to it automatically
                cls.log.DEBUG("Found a custom postprocessor")
                cls.processor = getattr(model_class, "postprocessor") 
            else:
                cls.log.DEBUG("Using default postprocessor")
                cls.processor = cls.default_postprocessor

            # Make a cuda context in main thread but then pop it out in order to push into worker thread
            cls.ctx = cuda.Device(0).make_context()
            cls.engine = ImageTensorRTInference()
            cls.engine.load_engine(path)
            cls.ctx.pop()
            cls.use_tensorrt = True

            
            return cls()

    @staticmethod
    def default_postprocessor(*input):
        return input
            

