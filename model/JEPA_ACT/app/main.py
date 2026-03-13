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

from utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0"],
    help="which devices to use on local machine",
)


def process(rank, fname, world_size, devices): 
    import os, sys

    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(":")[-1])
    
    from utils.logger import Logger
    
    logger = Logger()
        
    with open(fname, "r") as f:
        params = yaml.load(f, Loader = yaml.FullLoader)
        logger.INFO(f"Rank {rank} Loaded parameters")

    world_size, rank = init_distributed(rank_and_world_size = (rank, world_size))

    if rank == 0:
        Logger.set_levels("INFO", "ERROR", "WARNING", "DEBUG", "CUSTOM")
    else:
        Logger.set_levels("ERROR", "DEBUG", "CUSTOM")

        
    try:
        importlib.import_module(f"app.{params['app']}.train").main(params, fname)
    except KeyboardInterrupt:
        logger.ERROR(f"Keyboard Interrupt detected on {rank=}")
    except Exception as e:
        logger.ERROR(f"Error on {rank=}", full_traceback = e)
    finally:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        logger.DEBUG(f"Destroyed process group on {rank=}")

if __name__ == "__main__":
    args_parser = parser.parse_args()
        
    num_gps = len(args_parser.devices)
    mp.set_start_method("spawn")
    for rank in range(num_gps):
        mp.Process(target = process, args = (rank, args_parser.fname, num_gps, args_parser.devices)).start()