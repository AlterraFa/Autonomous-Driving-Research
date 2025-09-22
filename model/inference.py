import os, re
import time
import numpy as np
import threading
import importlib
import inspect

import cv2
import numpy as np

import torch
import torch.nn as nn

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from rich import print
from traceback import print_exc
if not hasattr(np, "float"): np.float = np.float64

torch.set_float32_matmul_precision("highest")

class ModelLoader:
    @staticmethod
    def _extract_class(path: str):
        """
        Extract the class object from a checkpoint path.

        Args:
            path (str): Path to checkpoint file 
                        (e.g. 'model/PilotNet/PilotNetExperiment/run5/best_PilotNetStatic_run5.pt')

        Returns:
            type: Class object found in the project structure
        """
        fname = os.path.basename(path)
        match = re.search(r"best_(.+?)_run\d+\.(?:pt|engine)$", fname)
        if not match:
            raise ValueError(f"Could not parse class name from filename: {fname}")
        class_name = match.group(1)

        dir_path = os.path.dirname(path)
        while dir_path and os.path.basename(dir_path) != "model":
            dir_path = os.path.dirname(dir_path)

        if not dir_path:
            raise RuntimeError("Could not find 'model' directory in path")

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
                    return getattr(module, class_name)
            except ModuleNotFoundError:
                continue

        raise ImportError(f"Could not find class '{class_name}' in any of: {candidate_modules}")

class AsyncInference:
    def __init__(self):
        self.input_data = None
        self.output_data = None
        
        self._event = threading.Event()
        self._lock = threading.Lock()
        self.infer_thread = threading.Thread(target=self._inference_torch if self.use_tensorrt == False else self._inference_tensorrt, daemon=True)
        self.infer_thread.start()

    def _inference_torch(self):

        self.pytorch = self._from_checkpoint(self.model_class, self.path, device = self.device)
        self.device = next(self.pytorch.parameters()).device

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
                output = self.pytorch(inp, processor_data).detach().cpu().numpy()[0]
                    
            with self._lock:
                self.output_data = output
        
    def _inference_tensorrt(self):

        if hasattr(self.model_class, "postprocessor"):
            self.processor = getattr(self.model_class, "postprocessor")
        else:
            self.processor = self.default_postprocessor

        self.ctx = cuda.Device(0).make_context()
        self.engine = ImageTensorRTInference()
        self.engine.load_engine(self.path)

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
                raw_output = self.engine.infer(inp)
                output = self.processor(raw_output, processor_data)
                print(raw_output)

            except Exception as e:
                print_exc()
                break

            with self._lock:
                self.output_data = output
        self.ctx.pop()

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
        fname = os.path.basename(path)
        name, ext = os.path.splitext(fname)
        
        cls.model_class = ModelLoader._extract_class(path)
        cls.path = path
        cls.device = device

        if ext == ".pt":
            cls.use_tensorrt = False
            return cls()
        elif ext == '.engine':
            cls.use_tensorrt = True

            
            return cls()

    @staticmethod
    def default_postprocessor(*input):
        return input
            


class TensorRTHelper:
    def __init__(self, *args, **kwargs):
        pass
    
    @staticmethod
    def _set_workspace(config, builder, gb: float):
        bytes_ = int(gb * (1 << 30))
        if hasattr(config, "set_memory_pool_limit"):  # TRT ≥ 8.6
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_)
        elif hasattr(config, "max_workspace_size"):   # TRT 8.0–8.5
            config.max_workspace_size = bytes_
        elif hasattr(builder, "max_workspace_size"):  # very old
            builder.max_workspace_size = bytes_
        else:
            raise RuntimeError("Cannot set workspace size on this ImageTensorRT build.")

    
    def _prepare_bindings(self):
        import tensorrt as trt
        print(f"[green][INFO][/]: Preparing engine bindings. TRT {trt.__version__}")
        self.stream = cuda.Stream()

        if hasattr(self.engine, "get_tensor_shape"):
            self.uses_io_tensors = True
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            modes = {n: self.engine.get_tensor_mode(n) for n in names}
            self.input_tensors  = [n for n in names if modes[n] == trt.TensorIOMode.INPUT]
            self.output_tensors = [n for n in names if modes[n] == trt.TensorIOMode.OUTPUT]

            # keep your preferred names if present
            if self.input_name not in self.input_tensors and self.input_tensors:
                self.input_name = self.input_tensors[0]
            if self.output_name not in self.output_tensors and self.output_tensors:
                self.output_name = self.output_tensors[0]
            return

        print(f"[green][INFO][/]: Preparing engine bindings. Fall back to TRT legacy version {trt.__version__}")
        self.uses_io_tensors = False
        self.bindings = [None] * self.engine.num_bindings
        self.input_idx  = self.engine.get_binding_index(self.input_name)
        self.output_idx = self.engine.get_binding_index(self.output_name)

    def _np_dtype_from_trt(self, dt):
        return {
            trt.DataType.FLOAT:  np.float32,
            trt.DataType.HALF:   np.float16,
            trt.DataType.INT8:   np.int8,
            trt.DataType.UINT8:  np.uint8 if hasattr(trt.DataType, "UINT8") else np.uint8,
            trt.DataType.INT32:  np.int32,
            trt.DataType.BOOL:   np.bool_,
        }.get(dt, np.float32)

    def _alloc_io(self, inp: np.ndarray):
        shape = tuple(inp.shape)

        if self.uses_io_tensors:
            if hasattr(self.context, "set_optimization_profile_async"):
                self.context.set_optimization_profile_async(0, self.stream.handle)

            self.context.set_input_shape(self.input_name, shape)
            in_bytes = int(inp.nbytes)
            if self.d_in is None or getattr(self, "_in_bytes", 0) != in_bytes:
                if self.d_in is not None: self.d_in.free()
                self.d_in = cuda.mem_alloc(int(in_bytes))
                self._in_bytes = in_bytes
            self.context.set_tensor_address(self.input_name, int(self.d_in))

            out_shapes = {}
            for name in self.output_tensors:
                shp = tuple(self.context.get_tensor_shape(name))
                dt_trt = self.engine.get_tensor_dtype(name)
                dt_np  = self._np_dtype_from_trt(dt_trt)
                nbytes = int(int(np.prod(shp)) * np.dtype(dt_np).itemsize)

                need_new = (
                    name not in self.out_ptrs or
                    self.out_shapes.get(name) != shp or
                    self.out_dtypes.get(name) != dt_np
                )
                if need_new:
                    if name in self.out_ptrs and self.out_ptrs[name] is not None:
                        self.out_ptrs[name].free()
                    self.out_ptrs[name] = cuda.mem_alloc(int(nbytes))
                    self.out_shapes[name] = shp
                    self.out_dtypes[name] = dt_np

                self.context.set_tensor_address(name, int(self.out_ptrs[name]))
                out_shapes[name] = shp

            return out_shapes

        self.context.set_binding_shape(self.input_idx, shape)
        out_shape = tuple(self.context.get_binding_shape(self.output_idx))
        in_bytes  = int(inp.nbytes)
        out_elems = int(np.prod(out_shape))
        out_bytes = int(out_elems * np.dtype(self.out_dtype).itemsize)
        if self.last_io_shape != (shape, out_shape):
            if self.d_in is not None:  self.d_in.free()
            if self.d_out is not None: self.d_out.free()
            self.d_in  = cuda.mem_alloc(in_bytes)
            self.d_out = cuda.mem_alloc(out_bytes)
            self.last_io_shape = (shape, out_shape)
        self.bindings[self.input_idx]  = int(self.d_in)
        self.bindings[self.output_idx] = int(self.d_out)
        return {self.output_name: out_shape}

    def get_engine_io_shapes(self, profile_index=0):
        import tensorrt as trt

        e = self.engine
        if e is None:
            raise RuntimeError("Engine not loaded")

        def dims_to_tuple(d):
            try:
                return tuple(int(x) for x in d)
            except Exception:
                if hasattr(d, "nbDims") and d.nbDims is not None and d.nbDims >= 0:
                    return tuple(int(d[i]) for i in range(d.nbDims))
                return None

        def infer_layout_and_hw(shape_tup):
            if not shape_tup or len(shape_tup) != 4:
                return None, None, None
            if shape_tup[1] in (1, 3):
                _, _, H, W = shape_tup[0], shape_tup[1], shape_tup[2], shape_tup[3]
                return "NCHW", H, W
            if shape_tup[-1] in (1, 3):
                _, H, W, _ = shape_tup[0], shape_tup[1], shape_tup[2], shape_tup[3]
                return "NHWC", H, W
            _, _, H, W = shape_tup[0], shape_tup[1], shape_tup[2], shape_tup[3]
            return "NCHW", H, W

        info = {}

        # ---- New explicit I/O API
        if hasattr(e, "get_tensor_shape"):
            names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
            for name in names:
                is_input = (e.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                dtype    = e.get_tensor_dtype(name)
                shape    = dims_to_tuple(e.get_tensor_shape(name))
                dyn      = bool(shape and any(d == -1 for d in shape))

                min_s = opt_s = max_s = shape

                if is_input and dyn and hasattr(e, "get_tensor_profile_shape"):
                    try:
                        pmin, popt, pmax = e.get_tensor_profile_shape(name, profile_index)
                        tm, to, tx = dims_to_tuple(pmin), dims_to_tuple(popt), dims_to_tuple(pmax)
                        if tm and to and tx:
                            min_s, opt_s, max_s = tm, to, tx
                    except Exception:
                        pass

                layout, H, W = infer_layout_and_hw(opt_s) if is_input else (None, None, None)
                info[name] = {
                    "is_input": is_input,
                    "dtype": dtype,
                    "min": min_s, "opt": opt_s, "max": max_s,
                    "layout": layout, "H": H, "W": W,
                }
            return info

        for i in range(e.num_bindings):
            name    = e.get_binding_name(i)
            is_input= e.binding_is_input(i)
            dtype   = e.get_binding_dtype(i)
            shape   = dims_to_tuple(e.get_binding_shape(i))
            dyn     = bool(shape and any(d == -1 for d in shape))

            min_s = opt_s = max_s = shape
            if is_input and dyn and hasattr(e, "get_profile_shape"):
                try:
                    pmin, popt, pmax = e.get_profile_shape(profile_index, i)
                    tm, to, tx = dims_to_tuple(pmin), dims_to_tuple(popt), dims_to_tuple(pmax)
                    if tm and to and tx:
                        min_s, opt_s, max_s = tm, to, tx
                except Exception:
                    pass

            layout, H, W = infer_layout_and_hw(opt_s) if is_input else (None, None, None)
            info[name] = {
                "is_input": is_input,
                "dtype": dtype,
                "min": min_s, "opt": opt_s, "max": max_s,
                "layout": layout, "H": H, "W": W,
            }
        return info

class ImageTensorRTInference(TensorRTHelper):
    """
    Minimal YOLO -> ONNX -> ImageTensorRT pipeline, with inference.
    Uses Ultralytics only to read .pt weights.
    Exports a pre-NMS ONNX, builds TRT with Python API, runs NumPy NMS.
    """

    def __init__(self, logger_severity=trt.Logger.WARNING):
        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.bindings = []
        self.input_idx = None
        self.output_idx = None
        self.input_name = "images"
        self.output_name = "preds"
        self.d_in = None
        self.d_out = None
        self.out_dtype = np.float32
        self.last_io_shape = None
        self.uses_io_tensors = False
        self.stream = None
        self.out_ptrs = {}
        self.out_shapes = {}
        self.out_dtypes = {}
        self.input_tensors = []
        self.output_tensors = []


    def load_engine(self, engine_path: str):
        print(f"[green][INFO][/]: Loading engine file...")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load engine.")
        self.context = self.engine.create_execution_context()
        self._prepare_bindings()

    def infer(self, imgs_bgr: np.ndarray):

        """
        inps : np.ndarray (H,W,3) or list of np.ndarray
            Single image or list of images (BGR).
        Returns:
            list of (boxes, confs, cls_ids) per image
        """
        assert self.engine is not None and self.context is not None
        
        # if isinstance(inps, (torch.Tensor)):
        #     raise TypeError("Please convert to np.ndarray via tensor.detach().cpu().numpy()")

        # if isinstance(inps, (np.ndarray, torch.Tensor)) and inps.ndim == 3:
        #     inps = [inps]
        if isinstance(imgs_bgr, np.ndarray) and imgs_bgr.ndim == 3:
            imgs_bgr = [imgs_bgr]
        batch_size = len(imgs_bgr)

        ims = []
        for im_bgr in imgs_bgr:
            ims.append(im_bgr)
        ims = np.stack(ims, axis=0)
        ims = np.ascontiguousarray(ims)

        out_shapes = self._alloc_io(ims)
        cuda.memcpy_htod(self.d_in, ims)
        if self.uses_io_tensors:
            ok = self.context.execute_async_v3(self.stream.handle)
        else:
            ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("ImageTensorRT execution failed")
        self.stream.synchronize()

        outs = {}
        if self.uses_io_tensors:
            for name, shp in out_shapes.items():
                host = np.empty(shp, dtype=self.out_dtypes[name])
                cuda.memcpy_dtoh(host, self.out_ptrs[name])
                outs[name] = host
        else:
            out = np.empty(next(iter(out_shapes.values())), dtype=self.out_dtype)
            cuda.memcpy_dtoh(out, self.d_out)
            outs[self.output_name] = out

        preds = outs.get(self.output_name, next(iter(outs.values())))
        if preds.dtype == np.float16:
            preds = preds.astype(np.float32)
        return preds
    

class ImageTensorRTExport(TensorRTHelper):
    """
    Minimal YOLO -> ONNX -> ImageTensorRT pipeline, with inference.
    Uses Ultralytics only to read .pt weights.
    Exports a pre-NMS ONNX, builds TRT with Python API, runs NumPy NMS.
    """

    def __init__(self, logger_severity=trt.Logger.WARNING):
        super().__init__(self)
        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.bindings = []
        self.input_idx = None
        self.output_idx = None
        self.d_in = None
        self.d_out = None
        self.out_dtype = np.float32
        self.last_io_shape = None
        self.uses_io_tensors = False
        self.stream = None
        self.out_ptrs = {}
        self.out_shapes = {}
        self.out_dtypes = {}
        self.input_tensors = []
        self.output_tensors = []

        self.input_name = "images"
        self.output_name = "preds"

    @staticmethod
    def load_model_from_checkpoint(path: str, device: str = "cpu", **model_kwargs):
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
        fname = os.path.basename(path)
        match = re.search(r"best_(.+?)_run\d+\.pt", fname)
        if not match:
            raise ValueError(f"Could not parse class name from filename: {fname}")
        class_name = match.group(1)

        dir_path = os.path.dirname(path)
        while dir_path and os.path.basename(dir_path) != "model":
            dir_path = os.path.dirname(dir_path)

        if not dir_path:
            raise RuntimeError("Could not find 'model' directory in path")

        rel_path = os.path.relpath(os.path.dirname(path), dir_path)  
        parts = rel_path.split(os.sep)

        candidate_modules = []
        for i in range(len(parts)):
            subpath = ".".join(parts[:i+1])
            candidate_modules.append(f"model.{subpath}.model")
        candidate_modules.append("model")

        cls = None
        for module_path in candidate_modules:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name) and inspect.isclass(getattr(module, class_name)):
                    cls = getattr(module, class_name)
                    break
            except ModuleNotFoundError:
                continue

        if cls is None:
            raise ImportError(f"Could not find class '{class_name}' in any of: {candidate_modules}")

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

    def export_onnx(self, net: torch.nn.Module, dummy_inp: tuple, onnx_path="yolo_raw.onnx",
                    img_size=(640, 640), opset=13, dynamic=True):
        H, W = img_size

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "preds":  {0: "batch", 1: "num_dets"}
            }

        print("[green][INFO][/]: Exporting onnx")
        try:
            torch.onnx.export(
                net, tuple(dummy_inp), onnx_path,
                input_names=[self.input_name],
                output_names=[self.output_name],
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes
            )
        except Exception as e:
            print(f"[red][ERROR][/]: Export failed")
            print_exc()
            
        print(f"[green][INFO][/]: Export complete. Path: {onnx_path}")
        return onnx_path

    def build_engine(self, onnx_path, engine_path="yolo_raw.engine",
                     fp16=True, workspace_gb=1.0,
                     min_shape=(1,3,320,320), opt_shape=(1,3,640,640), max_shape=(1,3,960,960)):
        
        print("[green][INFO][/]: Exporting engine")
        try:
            builder = trt.Builder(self.logger)
            EXPL = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(EXPL)
            parser  = trt.OnnxParser(network, self.logger)
            config  = builder.create_builder_config()

            # workspace (version-safe)
            self._set_workspace(config, builder, workspace_gb)

            if fp16 and builder.platform_has_fast_fp16:
                print("[yellow][WARNING]: Fallback to fp16 precision")
                config.set_flag(trt.BuilderFlag.FP16)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(parser.get_error(i))
                    raise RuntimeError("Failed to parse ONNX.")
                    

            print("[green][INFO]: Optimizing profile. This might take some time")
            profile = builder.create_optimization_profile()
            profile.set_shape(self.input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            engine = None
            if hasattr(builder, "build_serialized_network"):   # TRT ≥ 8.0 modern path (required on 8.6+/9.x)
                print(f"[yellow][WARNING]: Using TRT version >= 8.x: {trt.__version__}")
                plan = builder.build_serialized_network(network, config)
                if plan is None:
                    raise RuntimeError("build_serialized_network() returned None")
                engine = self.runtime.deserialize_cuda_engine(plan)
            elif hasattr(builder, "build_engine"):              # Some 8.x builds
                print("[yellow][WARNING]: Using TRT version 8.x")
                engine = builder.build_engine(network, config)
            elif hasattr(builder, "build_cuda_engine"):         # TRT 7.x and earlier
                print("[yellow][WARNING]: Using TRT version <= 7.x")
                engine = builder.build_cuda_engine(network)
            else:
                raise RuntimeError("No supported build method found on this ImageTensorRT version.")

            if engine is None:
                raise RuntimeError("Failed to build ImageTensorRT engine.")

            print("[green][INFO]: Saving engine to disk")
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())

            self.engine  = engine
            self.context = engine.create_execution_context()
            self._prepare_bindings()
        
        except Exception as e:
            print(f"[red][ERROR][/]: Export failed")
            print_exc()
            
        print(f"[green][INFO][/]: Export complete. Path: {engine_path}")
        return engine_path

    
    def export_and_build(self, weights_path: str = None,
                         *dummy_inp, 
                         img_size=(270, 150),
                         opset=13, dynamic=True, fp16=True,
                         min_shape = (1, 3, 270, 150),
                         opt_shape = (4, 3, 270, 150), 
                         max_shape = (8, 3, 270, 150),
                         workspace_gb = 1.0, 
                         device = 'cpu'):
        if hasattr(self, "weights_path") == False:
            if weights_path is None: raise RuntimeError("No weights path specified")
        else: weights_path = self.weights_path
        
        base, _ = os.path.splitext(weights_path)
        onnx_path   = base + ".onnx"
        engine_path = base + ".engine"
        

        net = self.load_model_from_checkpoint(weights_path, device = device)
        self.export_onnx(net, dummy_inp, onnx_path = onnx_path, img_size = img_size, opset = opset, dynamic = dynamic)
        self.build_engine(onnx_path, engine_path, fp16 = fp16, min_shape = min_shape, opt_shape = opt_shape, max_shape = max_shape, workspace_gb = workspace_gb)
        return onnx_path, engine_path