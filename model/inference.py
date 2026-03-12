# Helper group
import os, re
import time
import threading
import importlib
import inspect
import yaml

# Compute group
import cv2
import numpy as np
import difflib
from .tensor_engine import ImageTensorRTInference

# Cuda group
import pycuda.driver as cuda
import pycuda.autoinit

# Logging group
from traceback import print_exc
from .logger import Logger
if not hasattr(np, "float"): np.float = np.float64


class ModelLoader:
    def __init__(self):
        self.log = Logger()
        pass
    
    def _extract_class(self, path: str):
        """
        Extract the class object from a checkpoint path by searching python files
        recursively in directories from the checkpoint up to the 'model' root.

        Args:
            path (str): Path to checkpoint file 

        Returns:
            type: Class object found in the project structure
        """
        self.log.DEBUG(f"Automatically finding module for the specified model")
        
        fname = os.path.basename(path)
        match = re.search(r"(?:best|last)_([^_.]+)(?:_.*)?\.(?:pt|engine)$", fname)
        if not match:
            raise ValueError(f"Could not parse class name from filename: {fname}")
        class_name = match.group(1)

        abs_ckpt_path = os.path.abspath(path)
        cwd = os.getcwd() 
        
        current_search_dir = os.path.dirname(abs_ckpt_path)
        

        while True:
            for root, dirs, files in os.walk(current_search_dir):
                
                if '__pycache__' in dirs: dirs.remove('__pycache__')
                if '.git' in dirs: dirs.remove('.git')
                if '.vscode' in dirs: dirs.remove('.vscode')
                if '.venv' in dirs: dirs.remove('.venv')

                for py_file in files:
                    if not py_file.endswith(".py"):
                        continue

                    full_file_path = os.path.join(root, py_file)
                    try:
                        rel_path = os.path.relpath(full_file_path, cwd)
                        if rel_path.startswith(".."):
                            rel_path = rel_path.strip("../")
                            
                        module_path = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                        module = importlib.import_module(module_path)
                        
                        if hasattr(module, class_name) and inspect.isclass(getattr(module, class_name)):
                            found_class = getattr(module, class_name)
                            self.log.INFO(f"Found class [bold]{class_name}[/] in [bold]{module_path}[/]")
                            return found_class
                            
                    except (ImportError, AttributeError, Exception) as e:
                        print(e)
                        continue

            if os.path.basename(current_search_dir) == "model":
                break
            
            parent_dir = os.path.dirname(current_search_dir)
            if parent_dir == current_search_dir:
                break
            current_search_dir = parent_dir
        self.log.ERROR(f"Could not find class [bold]{class_name}[/] in directories up to 'model' root.", exit_code=12)

    def _extract_yaml(self, path: str) -> str:
        """
        Extract the path to ANY .yaml configuration file from a checkpoint path 
        by searching recursively in directories from the checkpoint up to the 'model' root.
        
        It returns the first .yaml file found, prioritizing files closest to the checkpoint.

        Args:
            path (str): Path to checkpoint file 

        Returns:
            str: Absolute path to the found yaml file
        """
        self.log.DEBUG(f"Automatically finding any configuration file for the specified model")
        
        abs_ckpt_path = os.path.abspath(path)
        current_search_dir = os.path.dirname(abs_ckpt_path)
        
        while True:
            for root, dirs, files in os.walk(current_search_dir):
                
                if '__pycache__' in dirs: dirs.remove('__pycache__')
                if '.git' in dirs: dirs.remove('.git')
                if '.vscode' in dirs: dirs.remove('.vscode')

                for file in files:
                    if file.endswith(".yaml") or file.endswith(".yml"):
                        full_file_path = os.path.join(root, file)
                        self.log.INFO(f"Found config [bold]{file}[/] in [bold]{root}[/]")
                        return full_file_path

            if os.path.basename(current_search_dir) == "model":
                break
            
            parent_dir = os.path.dirname(current_search_dir)
            if parent_dir == current_search_dir:
                break
            
            current_search_dir = parent_dir

        self.log.ERROR(f"Could not find any .yaml file in directories up to 'model' root.", exit_code=12)

    def _extract_argvals(self, cls, yaml_path, threshold = 0.8):

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        sig = inspect.signature(cls)
        required_args = []
        for name, param in sig.parameters.items():
            # if param.default is inspect.Parameter.empty and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            required_args.append(name)
        
        all_kv = {}
        def recurse(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    all_kv[k] = v
                    recurse(v)
            elif isinstance(d, list):
                for item in d:
                    recurse(item)
                    
        recurse(config)
        available_keys = list(all_kv.keys())
        results = {}
        
        for target in required_args:
            # Check for exact match first (case-insensitive)
            exact_match = next((k for k in available_keys if k.lower() == target.lower()), None)
            
            if exact_match:
                results[target] = all_kv[exact_match]
            else:
                # Perform fuzzy matching using difflib
                matches = difflib.get_close_matches(target, available_keys, n=1, cutoff=threshold)
                if matches:
                    results[target] = all_kv[matches[0]]
                # else:
                #     results[target] = None # Or a default value
                    
        return results

class AsyncInference:
    def __init__(self, path, device = "cpu", batch_output = False, **model_kwargs):
        import torch
        self.log = Logger()
        self.input_data = None
        self.output_data = None

        self.batch_output = batch_output

        self.load_model(path, device, **model_kwargs)
        
        self._event = threading.Event()
        self._lock = threading.Lock()
        self.infer_thread = threading.Thread(target=self._inference_torch if self.use_tensorrt == False else self._inference_tensorrt, daemon=True)
        self.infer_thread.start()
    
    def _inference_torch(self):
        import torch
        
        while not self._event.is_set():

            with self._lock:
                data = self.input_data
                self.input_data = None
            if data is None:
                time.sleep(0.05)
                continue

            try:  # ← ADD TRY-CATCH
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
                    processor_data = None  # ← FIX THE ELLIPSIS

                with torch.no_grad():
                    if processor_data != None:
                        output = self.pytorch(*inp, processor_data)
                    else:
                        output = self.pytorch(*inp)
                
                    if isinstance(output, torch.Tensor):
                        # Single tensor output
                        if self.batch_output == True:
                            output = output.detach().cpu().numpy()[0]
                        else:
                            output = output.detach().cpu().numpy()[0, 0]
                    elif isinstance(output, (tuple, list)):
                        # Multiple tensor outputs (tuple or list)
                        if self.batch_output:
                            output = tuple(
                                tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
                                for tensor in output
                            )
                        else:
                            output = tuple(
                                tensor.detach().cpu().numpy()[0] if isinstance(tensor, torch.Tensor) else tensor
                                for tensor in output
                            )
                        
                with self._lock:
                    self.output_data = output
                    
            except Exception as e:  # ← ADD EXCEPTION HANDLING
                self.log.ERROR(f"PyTorch inference error: {e}")
                print_exc()
                break
                
        self.log.INFO("PyTorch inference stopped")  # ← ADD LOG MESSAGE
        
    def _inference_tensorrt(self):

        import torch
        self.ctx.push()
        self.log.INFO("Started engine inference")

        try:
            while not self._event.is_set():
                with self._lock:
                    data = self.input_data
                    self.input_data = None   # consume once
                if data is None:
                    time.sleep(0.075)        # yield CPU, avoid busy spin
                    continue

                inp_img, extra_data = data
                
                if isinstance(inp_img, (torch.Tensor, np.ndarray, cv2.Mat)):
                    inp = [np.ascontiguousarray(inp_img)]
                else:
                    inp = [np.ascontiguousarray(img) for img in inp_img]

                if not isinstance(extra_data, (torch.Tensor, np.ndarray, cv2.Mat)):
                    processor_data = extra_data
                else:
                    ...
                
                raw_output = self.engine.infer(*inp)
                output = self.processor(raw_output, processor_data)

                with self._lock:
                    self.output_data = output

        except Exception as e:
            print_exc()
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
    def _from_checkpoint(cls, path: str, cfg_path: str, device: str = "cpu", **model_kwargs):
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
        import torch

        # Load checkpoint
        try:
            state_dict = torch.load(path, map_location=device)
            if isinstance(state_dict, dict):

                required_kwargs = ModelLoader()._extract_argvals(cls, cfg_path, 0.7)
                print(required_kwargs)
                
                model = cls(**required_kwargs, **model_kwargs)
                model.load_state_dict(state_dict)
                model.to(device).eval()
                return model
        except Exception:
            torch.serialization.add_safe_globals([cls])
            model = torch.load(path, weights_only=False, map_location=device)
            model.to(device).eval()
            return model
    
    def load_model(self, path: str, device: str = "cpu", **model_kwargs):
        """
        Load a model checkpoint, automatically inferring class name and module path from filename.

        Args:
            path (str): Path to checkpoint file (e.g. 'model/PilotNet/best_PilotNetStatic_run1.pt')
            device (str): Device to load model onto ('cpu' or 'cuda')
            **model_kwargs: Extra arguments to pass to the model constructor (needed if state_dict only)

        Returns:
            torch.nn.Module: Loaded model
        """

        basepath = os.path.dirname(path)
        fname = os.path.basename(path)
        name, ext = os.path.splitext(fname)
        
        model_class = ModelLoader()._extract_class(path)
        cfg_path    = ModelLoader()._extract_yaml(path)
        if cfg_path is not None: model_class.config_path = cfg_path
        
        pt_path = basepath + "/" + name + ".pt"

        if ext == ".pt":
            self.log.INFO("Using Dynamic graph model variant (Pytorch)")
            self.pytorch = self._from_checkpoint(model_class, pt_path, cfg_path, device = device)
            self.device = next(self.pytorch.parameters()).device
            self.use_tensorrt = False
        elif ext == '.engine':
            self.log.INFO("Using Engine variant")
            if hasattr(model_class, "postprocessor"):
                self.log.DEBUG("Found a custom postprocessor")
                self.processor = getattr(model_class, "postprocessor") 
            else:
                self.log.DEBUG("Using default postprocessor")
                self.processor = self.default_postprocessor

            self.pytorch = self._from_checkpoint(model_class, pt_path, cfg_path, device = 'cpu')
            # Make a cuda context in main thread but then pop it out in order to push into worker thread
            self.ctx = cuda.Device(0).make_context()
            self.engine = ImageTensorRTInference()
            self.engine.load_engine(path)
            self.ctx.pop()
            self.use_tensorrt = True

    @staticmethod
    def default_postprocessor(*input):
        return input
