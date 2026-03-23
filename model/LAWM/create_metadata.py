import os
import pandas as pd
import glob

def create_csv(directory, name, save_path = "."):
    recording_dirs = glob.glob(directory)
    os.makedirs(save_path, exist_ok = True)
    for record in recording_dirs:
        seq_dirs = glob.glob(os.path.join(record, "*"))
        seq_dirs = sorted(seq_dirs, key = lambda x: int(x.split("/")[-1][4:]))
        
        df = pd.DataFrame(
            seq_dirs
        )
        dset_name, ext = name.split(".")
        record_name = dset_name + "_" + os.path.basename(record) + "." + ext
        record_path = os.path.join(save_path, record_name)
        df.to_csv(record_path)
        
    
create_csv("../Autonomous_Dataset/jepa_probe/*", "Carla.csv", "csv_metadata/probe")
