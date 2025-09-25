import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
import time
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*legacy TorchScript-based ONNX export.*",
    category=DeprecationWarning,
)

from tqdm.auto import tqdm
from rich import print
from utils.messages.logger import Logger
if not hasattr(np, "float"): np.float = np.float64

class TensorRTHelper:
    def __init__(self, *args, **kwargs):
        self.log = Logger()
        self.input_names = ["img"]
        self.output_names = ["preds"]

        # self.d_in  = None
        # self.d_out = None
        self.d_ins  = []
        self.d_outs = []

        self.d_ins_size = []
    
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
        self.log.WARNING(f"Preparing engine bindings. TRT {trt.__version__}")
        self.stream = cuda.Stream()

        if hasattr(self.engine, "get_tensor_shape"):
            self.uses_io_tensors = True
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            modes = {n: self.engine.get_tensor_mode(n) for n in names}
            self.input_names  = [n for n in names if modes[n] == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in names if modes[n] == trt.TensorIOMode.OUTPUT]

            # keep your preferred names if present
            if self.input_names not in self.input_names and self.input_names:
                self.input_names = self.input_names
            if self.output_names not in self.output_names and self.output_names:
                self.output_names = self.output_names
            self.log.CUSTOM("SUCCESS", "Binding completed")
            return

        self.log.WARNING(f"Preparing engine bindings. Fall back to TRT legacy version {trt.__version__}")
        self.uses_io_tensors = False
        self.bindings = [None] * self.engine.num_bindings
        self.input_idx  = self.engine.get_binding_index(self.input_names)
        self.output_idx = self.engine.get_binding_index(self.output_names)
        self.log.CUSTOM("SUCCESS", "Binding completed")

    def _np_dtype_from_trt(self, dt):
        return {
            trt.DataType.FLOAT:  np.float32,
            trt.DataType.HALF:   np.float16,
            trt.DataType.INT8:   np.int8,
            trt.DataType.UINT8:  np.uint8,
            trt.DataType.INT32:  np.int32,
            trt.DataType.BOOL:   np.bool_,
        }.get(dt, np.float32)

    def _alloc_io(self, *inp_arr: np.ndarray):
        """Memory allocations for engine input and output on GPU"""
        
        if self.uses_io_tensors: # Newer TRT
            if hasattr(self.context, "set_optimization_profile_async"):
                self.context.set_optimization_profile_async(0, self.stream.handle)

            # Allocation for input
            for idx, (name, inp) in enumerate(zip(self.input_names, inp_arr)):
                self.context.set_input_shape(name, inp.shape)
                in_bytes = int(inp.nbytes)
                # Allocate if infer first time
                if idx + 1 > len(self.d_ins):
                    self.d_ins      += [cuda.mem_alloc(in_bytes)]
                    self.d_ins_size += [in_bytes]

                elif self.d_ins_size[idx] < in_bytes:
                    self.d_ins[idx].free()
                    self.d_ins[idx]      = cuda.mem_alloc(in_bytes)
                    self.d_ins_size[idx] = in_bytes
                self.context.set_tensor_address(name, self.d_ins[idx])

            # Allocation for output
            out_shapes = {}
            for name in self.output_names:
                shp = tuple(self.context.get_tensor_shape(name)) # get output shape
                dt_trt = self.engine.get_tensor_dtype(name) # get output data type
                dt_np  = self._np_dtype_from_trt(dt_trt) # Convert data type to numpy
                nbytes = int(int(np.prod(shp)) * np.dtype(dt_np).itemsize) # compute amount of bytes needed

                need_new = (
                    name not in self.out_ptrs or
                    self.out_shapes.get(name) != shp or
                    self.out_dtypes.get(name) != dt_np
                ) 
                if need_new:
                    if name in self.out_ptrs and self.out_ptrs[name] is not None:
                        self.out_ptrs[name].free() # Free the pointers from previous inference
                    self.out_ptrs[name] = cuda.mem_alloc(int(nbytes))
                    self.out_shapes[name] = shp
                    self.out_dtypes[name] = dt_np

                self.context.set_tensor_address(name, int(self.out_ptrs[name]))
                out_shapes[name] = shp

            return out_shapes

        # self.context.set_binding_shape(self.input_idx, shape)
        # out_shape = tuple(self.context.get_binding_shape(self.output_idx))
        # in_bytes  = int(inp.nbytes)
        # out_elems = int(np.prod(out_shape))
        # out_bytes = int(out_elems * np.dtype(self.out_dtype).itemsize)
        # if self.last_io_shape != (shape, out_shape):
        #     if self.d_in is not None:  self.d_in.free()
        #     if self.d_out is not None: self.d_out.free()
        #     self.d_in  = cuda.mem_alloc(in_bytes)
        #     self.d_out = cuda.mem_alloc(out_bytes)
        #     self.last_io_shape = (shape, out_shape)
        # self.bindings[self.input_idx]  = int(self.d_in)
        # self.bindings[self.output_idx] = int(self.d_out)
        # return {self.output_names: out_shape}
    
    @staticmethod
    def dims_to_tuple(d):
        try:
            return tuple(int(x) for x in d)
        except Exception:
            if hasattr(d, "nbDims") and d.nbDims is not None and d.nbDims >= 0:
                return tuple(int(d[i]) for i in range(d.nbDims))
            return None

    @staticmethod
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

    def get_engine_io_shapes(self, profile_index=0, verbose=False):
        import tensorrt as trt

        e = self.engine
        if e is None:
            raise RuntimeError("Engine not loaded")

        info = {}

        # ---- New explicit I/O API
        if hasattr(e, "get_tensor_shape"):
            names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
            for name in names:
                is_input = (e.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                dtype    = e.get_tensor_dtype(name)
                shape    = self.dims_to_tuple(e.get_tensor_shape(name))
                dyn      = bool(shape and any(d == -1 for d in shape))

                min_s = opt_s = max_s = shape
                if is_input and dyn and hasattr(e, "get_tensor_profile_shape"):
                    try:
                        tm, to, tx = list(map(self.dims_to_tuple, e.get_tensor_profile_shape(name, profile_index)))
                        if tm and to and tx:
                            min_s, opt_s, max_s = tm, to, tx
                    except Exception:
                        self.log.WARNING(f"Could not fetch profile shapes for {name}")

                layout, H, W = self.infer_layout_and_hw(opt_s) if is_input else (None, None, None)
                info[name] = {
                    "dtype": dtype,
                    "min": min_s, "opt": opt_s, "max": max_s,
                    "layout": layout, "H": H, "W": W,
                }
        else:
            for i in range(e.num_bindings):
                name     = e.get_binding_name(i)
                is_input = e.binding_is_input(i)
                dtype    = e.get_binding_dtype(i)
                shape    = self.dims_to_tuple(e.get_binding_shape(i))
                dyn      = bool(shape and any(d == -1 for d in shape))

                min_s = opt_s = max_s = shape
                if is_input and dyn and hasattr(e, "get_profile_shape"):
                    try:
                        tm, to, tx = list(map(self.dims_to_tuple, e.get_profile_shape(profile_index, i)))
                        if tm and to and tx:
                            min_s, opt_s, max_s = tm, to, tx
                    except Exception:
                        pass

                layout, H, W = self.infer_layout_and_hw(opt_s) if is_input else (None, None, None)
                info[name] = {
                    "dtype": dtype,
                    "min": min_s, "opt": opt_s, "max": max_s,
                    "layout": layout, "H": H, "W": W,
                }

        # --- Split into two dicts
        inputs  = {n: m for n, m in info.items() if e.get_tensor_mode(n) == trt.TensorIOMode.INPUT} \
                if hasattr(e, "get_tensor_shape") else \
                {n: m for n, m in info.items() if e.binding_is_input(list(info.keys()).index(n))}
        outputs = {n: m for n, m in info.items() if n not in inputs}

        # ----- Verbose printing
        if verbose:
            def print_table(title, subset):
                print(f"[{title}]")
                print(f"{'Name':20} {'DType':15} {'Min':20} {'Opt':20} {'Max':20} {'Layout':6} {'HxW'}")
                print("-" * 110)
                for name, meta in subset.items():
                    print(f"{name:20} {str(meta['dtype']):10} "
                        f"{str(meta['min']):20} {str(meta['opt']):20} {str(meta['max']):20} "
                        f"{str(meta['layout'] or ''):6} {str((meta['H'], meta['W']))}")
                print()

            if inputs:
                print_table("Inputs", inputs)
            if outputs:
                print_table("Outputs", outputs)

        return inputs, outputs


class ImageTensorRTInference(TensorRTHelper):
    def __init__(self, logger_severity=trt.Logger.WARNING):
        super().__init__(self)
        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.bindings = []
        self.d_ins  = []
        self.d_outs = []
        self.out_dtype = np.float32
        self.uses_io_tensors = False
        self.stream = None
        self.out_ptrs = {}
        self.out_shapes = {}
        self.out_dtypes = {}
        self.log = Logger()


    def load_engine(self, engine_path: str):
        self.log.DEBUG("Loading engine file...")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            self.log.ERROR("Failed to load engine. Either path to engine is incorrect or the engine is corrupted", exit_code = 16)
        self.context = self.engine.create_execution_context()
        self._prepare_bindings()
        self.log.CUSTOM('SUCCESS', "Engine loaded successfully. Prepare for inference.")

    def infer(self, *inp_imgs: np.ndarray):

        """
        inps : np.ndarray (H,W,3) or list of np.ndarray
            Single image or list of images (BGR).
        Returns:
            list of (boxes, confs, cls_ids) per image
        """
        assert self.engine is not None and self.context is not None
        
        for idx, imgs in enumerate(inp_imgs):
            if not isinstance(imgs, np.ndarray):
                raise TypeError(f"Please convert to input of type {type(imgs)} index {idx} to np.ndarray via tensor.detach().cpu().numpy()")

        out_shapes = self._alloc_io(*inp_imgs)
        for idx, inp in enumerate(inp_imgs):
            cuda.memcpy_htod_async(self.d_ins[idx], np.ascontiguousarray(inp), self.stream)
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
        # else:
        #     out = np.empty(next(iter(out_shapes.values())), dtype=self.out_dtype)
        #     cuda.memcpy_dtoh(out, self.d_out)
        #     outs[self.output_names] = out
        
        for name in self.output_names:
            outs.get(name)
            if outs[name].dtype == np.float16:
                outs[name] = outs[name].astype(np.float32)
        
        return outs


class ImageTensorRTExport(TensorRTHelper):
    def __init__(self, logger_severity=trt.Logger.WARNING):
        super().__init__(self)
        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None

        self.log = Logger()

    def load_model_from_checkpoint(self, path: str, device: str = "cpu", **model_kwargs):
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
        from model.inference import ModelLoader

        cls = ModelLoader()._extract_class(path)
        self.device = torch.device(device)
        # WARNING: I need to change this to support Pytorch<2 for Jetson Nano
        try:
            state_dict = torch.load(path, map_location=self.device)
            if isinstance(state_dict, dict):
                model = cls(**model_kwargs)
                model.load_state_dict(state_dict)
                model.to(self.device).eval()
                if not hasattr(model, "input_metadata"):
                    self.log.ERROR("The model does not have `input_metadata`", exit_code = 12)
                self.net = model
        except Exception:
            torch.serialization.add_safe_globals([cls]) if hasattr(torch.serialization, "add_safe_global") else None
            # Torch < 2.0 does not have weights_only arg
            if "weights_only" in torch.load.__code__.co_varnames:
                model = torch.load(path, weights_only=False, map_location=self.device)
            else:
                model = torch.load(path, map_location=self.device)
            model.to(self.device).eval()
            if not hasattr(model, "input_metadata"):
                self.log.ERROR("The model does not have `input_metadata`")
            self.net = model

    def export_onnx(self, onnx_path="model.onnx",
                    opset=13, dynamic=True):

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {}
            dynamic_axes = {name: {0: "batch"} for name in self.net.input_metadata.keys()}

        dummy_inp        = []
        self.input_names = []
        for key, value in self.net.input_metadata.items():
            dummy_inp        += [torch.zeros(value)]
            self.input_names += [key]
        
        # Check if there is an output name in the model.
        if not hasattr(self.net, "output_names"):
            self.log.WARNING("The model does not have `output_names`. This is optional.")
        else:
            self.output_names = self.net.output_names
        
        self.log.INFO("Exporting onnx")
        try:
            torch.onnx.export(
                self.net, tuple(dummy_inp), onnx_path,
                input_names   = self.input_names,
                output_names  = self.output_names,
                opset_version = opset,
                do_constant_folding = True,
                dynamic_axes = dynamic_axes,
            )
        except Exception as e:
            self.log.ERROR("Export failed", exit_code = 13, full_traceback = e)
        
        # self.net.to(self.device) # Moving back and forth between cuda and cpu is the root cause of slow engine build. But why the fuck???
        self.log.CUSTOM("SUCCESS", f"Export complete. Path: {onnx_path}")
        return onnx_path

    def build_engine(self, onnx_path, engine_path,
                     fp16 = True, workspace_gb = 1.0,
                     min_batch = 1, opt_batch = 1, max_batch = 1):
        
        self.log.INFO("Exporting engine.")
        try:
            builder = trt.Builder(self.logger)
            EXPL = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(EXPL)
            parser  = trt.OnnxParser(network, self.logger)
            config  = builder.create_builder_config()

            # workspace (version-safe)
            self._set_workspace(config, builder, workspace_gb)

            if fp16 and builder.platform_has_fast_fp16:
                self.log.INFO("Quantizing to fp16 precision.")
                config.set_flag(trt.BuilderFlag.FP16)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(parser.get_error(i))
                    self.log.ERROR("Failed to parse ONNX.", exit_code = 13)
                    

            self.log.INFO("Optimizing profile. This might take some time...")
            profile = builder.create_optimization_profile()
            for name in self.input_names:
                base_shape = list(self.net.input_metadata[name])  # e.g. (1, 3, 150, 270)

                # Copy and replace batch dim
                min_shape = tuple([min_batch] + base_shape[1:])
                opt_shape = tuple([opt_batch] + base_shape[1:])
                max_shape = tuple([max_batch] + base_shape[1:])
                profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            engine = None
            if hasattr(builder, "build_serialized_network"):   # TRT ≥ 8.0 modern path (required on 8.6+/9.x)
                self.log.DEBUG(f"Using TRT version >= 8.x: {trt.__version__}")
                plan = builder.build_serialized_network(network, config)
                if plan is None:
                    raise RuntimeError("build_serialized_network() returned None")
                engine = self.runtime.deserialize_cuda_engine(plan)
            elif hasattr(builder, "build_engine"):              # Some 8.x builds
                self.log.DEBUG(f"Using TRT version 8.x")
                engine = builder.build_engine(network, config)
            elif hasattr(builder, "build_cuda_engine"):         # TRT 7.x and earlier
                self.log.DEBUG("Using TRT version <= 7.x")
                engine = builder.build_cuda_engine(network)
            else:
                self.log.ERROR("No supported build method found on this ImageTensorRT version.", exit_code = 13)

            if engine is None:
                self.log.ERROR("Failed to build ImageTensorRT engine.", exit_code = 13)

            self.log.INFO("Saving engine to disk.")
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())

            self.engine  = engine
            self.context = engine.create_execution_context()
            self._prepare_bindings()
        
        except Exception as e:
            self.log.ERROR("Export failed.", full_traceback = e, exit_code = 14)
            
        self.log.CUSTOM("SUCCESS", f"Export complete. Path: {engine_path}")
        return engine_path

    
    def export_and_build(self, weights_path: str = None,
                         opset=16, dynamic=True, fp16=True,
                         min_batch = 1,
                         opt_batch = 1, 
                         max_batch = 1,
                         workspace_gb = 1.0, 
                         device = 'cpu'):
        if hasattr(self, "weights_path") == False:
            if weights_path is None: self.log.ERROR("No weights path specified", exit_code = 15)
        else: weights_path = self.weights_path

        def ensure_relative(path: str) -> str:
            if not path.startswith("./"):
                return "./" + path
            return path
        
        weights_path = ensure_relative(weights_path)
        base, _ = os.path.splitext(weights_path)
        onnx_path   = base + ".onnx"
        engine_path = base + ".engine"
        

        self.load_model_from_checkpoint(weights_path, device = device)
        self.export_onnx(onnx_path = onnx_path, opset = opset, dynamic = dynamic)
        self.build_engine(onnx_path, engine_path, fp16 = fp16, min_batch = min_batch, opt_batch = opt_batch, max_batch = max_batch, workspace_gb = workspace_gb)
        return onnx_path, engine_path

if __name__ == "__main__":
    import argparse
    log = Logger()

    parser = argparse.ArgumentParser(
        description="Export a PyTorch model to ONNX and build a TensorRT engine."
    )
    
    parser.add_argument(
        "--model-path", "-P",
        required=True,
        help="Path to the trained PyTorch model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--batch", "-B",
        default="1.1.1",
        help="Batch profile in the format 'min.opt.max' (e.g., 1.4.8). "
             "min=minimum batch size, opt=optimized batch size, max=maximum batch size."
    )
    parser.add_argument(
        "--workspace-gb",
        default=1.0,
        type=float,
        help="Workspace memory limit for TensorRT in GB (default: 1.0)."
    )
    parser.add_argument(
        "--use-fp16", "-fp16",
        action="store_true",
        help="Enable FP16 precision if supported by TensorRT."
    )
    parser.add_argument(
        "--test-iter",
        "-iter",
        type = int,
        default = 1,
        help="Number of iteration during speed up testing"
    )
    
    args = parser.parse_args()
    
    # --- Parse batch string ---
    try:
        parts = args.batch.split(".")
        if len(parts) != 3:
            raise ValueError
        min_batch, opt_batch, max_batch = map(int, parts)
    except Exception:
        raise ValueError(
            f"Invalid format for --batch '{args.batch}'. "
            "Expected format: 'min.opt.max' (e.g., 1.4.8)."
        )
    
    # --- Call exporter ---
    exporter = ImageTensorRTExport()
    _, engine_path = exporter.export_and_build(
        args.model_path, 
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        workspace_gb=args.workspace_gb, 
        fp16=args.use_fp16,
        device = 'cpu'
    )
    exporter.net = exporter.net.to('cuda')
    
    inference = ImageTensorRTInference()
    inference.load_engine(engine_path)
    log.INFO("I/O engine INFO")
    inputs_info, outputs_info = inference.get_engine_io_shapes(verbose = True)
    
    for profile_key in ["min", "opt", "max"]:
        log.INFO(f"Checking for output discrepancy on profile: {profile_key}")

        # build dummy inputs at this profile size
        dummy_inputs = []; dummy_inputs_metadata = {}
        for name, shapes in inputs_info.items():
            shape_data = shapes[profile_key]
            dummy_inputs.append(np.zeros(shape_data, dtype=np.float32))  # dtype can match inputs_info[name]['dtype']
            dummy_inputs_metadata.update({name: torch.tensor(dummy_inputs[-1]).to("cuda")})
            

        # run inference
        dummy_outputs = inference.infer(*dummy_inputs)

        # compare output shapes
        for name, value in dummy_outputs.items():
            expected = outputs_info[name][profile_key]
            if tuple(value.shape) != tuple(expected):
                log.ERROR(
                    f"Output discrepancy detected at [{name}] "
                    f"(profile={profile_key}, expected={expected}, got={value.shape})",
                    exit_code = 20
                )
        log.CUSTOM("SUCCESS", "No discrepancy found")
    
        log.INFO(f"Testing inference speedup by using TensorRT on profile {profile_key}")
        exporter.net(**dummy_inputs_metadata)
        

        # --- TensorRT benchmark ---
        start_trt = time.perf_counter()
        for _ in tqdm(range(args.test_iter), desc=f"TensorRT [{profile_key}]", ncols = 150):
            inference.infer(*dummy_inputs)
        torch.cuda.synchronize()  # make sure GPU work is done
        end_trt = time.perf_counter()
        trt_time = (end_trt - start_trt) / args.test_iter

        # --- PyTorch benchmark ---
        start_torch = time.perf_counter()
        for _ in tqdm(range(args.test_iter), desc=f"PyTorch [{profile_key}]", ncols = 150):
            with torch.no_grad():
                exporter.net(**dummy_inputs_metadata)
        torch.cuda.synchronize()
        end_torch = time.perf_counter()
        torch_time = (end_torch - start_torch) / args.test_iter

        # --- results ---
        speedup = torch_time / trt_time if trt_time > 0 else float("inf")
        log.INFO(
            f"[{profile_key}] PyTorch avg: {torch_time*1e3:.3f} ms | "
            f"TensorRT avg: {trt_time*1e3:.3f} ms | "
            f"Speedup: [bold][cyan]{speedup:.2f}[/][/]x"
        )
        print()