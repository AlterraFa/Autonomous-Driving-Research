import os, sys
import resource
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import argparse
import multiprocessing as mp
import yaml
import importlib
import torch
import glob
import re
from pathlib import Path

from utils.distributed import init_distributed
from app.probe_notebook import extract_nb_params

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--continue",
    dest="continue_path",
    type=str,
    default=None,
    help="path to a previous run directory (e.g. ./Experiment/probe/run1 or ./Experiment/run1)",
)
parser.add_argument(
    "--probe-notebook",
    dest="probe_notebook",
    type=str,
    default=None,
    help="path to latent-probe.ipynb; when set, extract PARAMS",
)


def _resolve_continue_run_dir(continue_path: str) -> str:
    raw_path = os.path.abspath(os.path.expanduser(continue_path))

    if os.path.isdir(raw_path):
        if os.path.basename(raw_path) == "weights":
            return os.path.dirname(raw_path)
        return raw_path

    run_name = os.path.basename(raw_path)
    run_name_match = re.fullmatch(r"run\d+", run_name)
    if run_name_match:
        base_dir = os.path.dirname(raw_path)
        candidates = [
            os.path.join(base_dir, mode, run_name)
            for mode in ("action", "probe", "pretraining", "straightening")
            if os.path.isdir(os.path.join(base_dir, mode, run_name))
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous continue path '{continue_path}'. Matches: {candidates}. "
                "Please pass the full run path including mode folder."
            )

    raise FileNotFoundError(
        f"Could not resolve continue run directory from '{continue_path}'."
    )


def _find_run_yaml(run_dir: str) -> str:
    yaml_candidates = sorted(glob.glob(os.path.join(run_dir, "*.yaml")))
    yaml_candidates.extend(sorted(glob.glob(os.path.join(run_dir, "*.yml"))))
    if not yaml_candidates:
        raise FileNotFoundError(
            f"No YAML config found in run directory: {run_dir}"
        )
    return yaml_candidates[0]


def process(rank, fname, world_size, devices, continue_path=None, probe_notebook=None): 
    import os, sys

    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(":")[-1])
    
    from utils.logger import Logger
    
    logger = Logger()

    config_path = fname
    resolved_continue_dir = None
    nb_probed = bool(probe_notebook)

    if nb_probed:
        extracted = extract_nb_params(probe_notebook)
        params = extracted["PARAMS"]
        config_path = str(Path(probe_notebook).resolve())
        logger.INFO(f"Loaded PARAMS from probe notebook: {config_path}")

        notebook_continue_path = extracted.get("CONTINUE_PATH")
        effective_continue_path = continue_path or notebook_continue_path
        if effective_continue_path:
            resolved_continue_dir = _resolve_continue_run_dir(str(effective_continue_path))
            logger.INFO(f"Continuing from run directory: {resolved_continue_dir}")
    elif continue_path:
        resolved_continue_dir = _resolve_continue_run_dir(continue_path)
        config_path = _find_run_yaml(resolved_continue_dir)
        logger.INFO(f"Continuing from run directory: {resolved_continue_dir}")
        logger.INFO(f"Loading run config from: {config_path}")

    if not nb_probed:
        with open(config_path, "r") as f:
            params = yaml.load(f, Loader = yaml.FullLoader)
            logger.INFO(f"Rank {rank} Loaded parameters")

    if resolved_continue_dir is not None:
        meta_cfg = params.setdefault("meta", {})
        meta_cfg.pop("continue_train", None)
        meta_cfg.pop("continue_from_run", None)
        meta_cfg["continue_from_path"] = resolved_continue_dir

    world_size, rank = init_distributed(rank_and_world_size = (rank, world_size))

    if rank == 0:
        Logger.set_levels("INFO", "ERROR", "WARNING", "DEBUG", "CUSTOM")
    else:
        Logger.set_levels("ERROR", "DEBUG", "CUSTOM")

    try:
        task_module = "train"
        importlib.import_module(f"app.{params['app']}.{task_module}").main(params, config_path)
    except KeyboardInterrupt:
        logger.ERROR(f"Keyboard Interrupt detected on {rank=}")
    except Exception as e:
        logger.ERROR(f"Error on {rank=}", full_traceback = e)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            logger.DEBUG(f"Destroyed process group on {rank=}")

if __name__ == "__main__":
    args_parser = parser.parse_args()

    num_gps = len(args_parser.devices)
    devices = args_parser.devices

    mp.set_start_method("spawn")
    for rank in range(num_gps):
        mp.Process(
            target=process,
            args=(
                rank,
                args_parser.fname,
                num_gps,
                devices,
                args_parser.continue_path,
                args_parser.probe_notebook,
            ),
        ).start()
