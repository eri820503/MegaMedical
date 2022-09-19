import numpy as np 
import submitit 
import math 
import matplotlib.pyplot as plt 
import os
import pandas as pd
pd.set_option('display.max_rows',100)

# Megamedical imports
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
import megamedical.utils as utils
              
        
def process_dataset(datasets,
                    subdsets=None,
                    save=False,
                    slurm=False, 
                    visualize=False,
                    redo_processed=True,
                    resolutions=[64, 128, 256],
                    show_hists=False,
                    version="4.0",
                    timeout=540,
                    mem_gb=32):
    assert not (len(datasets) > 1 and visualize), "Can't visualize a list of processing."
    assert not (slurm and visualize), "If you are submitting slurm no vis."
    
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])
    
    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        subdset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for subdset in subdset_names:
            if slurm:
                slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{subdset}")
                executor = submitit.AutoExecutor(folder=slurm_root)
                executor.update_parameters(timeout_min=timeout, mem_gb=mem_gb, slurm_partition="sablab", slurm_wckey="")
                job = executor.submit(do.proc_func,
                                      subdset,
                                      pps.produce_slices,
                                      True,
                                      False,
                                      version,
                                      visualize,
                                      save,
                                      show_hists,
                                      resolutions,
                                      redo_processed)
            else:
                do.proc_func(subdset,
                             pps.produce_slices,
                             True,
                             False,
                             version,
                             visualize,
                             save,
                             show_hists,
                             resolutions,
                             redo_processed)
                

def generate_label_info_files(datasets,
                              subdsets=None,
                              save=False,
                              slurm=False,
                              resolutions=[64, 128, 256],
                              version="4.0",
                              timeout=180,
                              mem_gb=16,
                              volume_wide=True):
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        subdset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for subdset in subdset_names:
            if slurm:
                slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{subdset}")
                executor = submitit.AutoExecutor(folder=slurm_root)
                executor.update_parameters(timeout_min=timeout, mem_gb=mem_gb, slurm_partition="sablab", slurm_wckey="")
                job = executor.submit(pps.label_info,
                                      do,
                                      subdset,
                                      version,
                                      resolutions,
                                      save)
            else:
                pps.label_info(do,
                               subdset,
                               version,
                               resolutions,
                               save)
                

def get_processing_status(datasets,
                          version="4.0"):
    
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])
        
    dataset_objects = [utils.build_dataset(ds) for ds in datasets]
    
    dp_objects = []
    for do in dataset_objects:
        for subset in do.dset_info.keys():
            for modality in do.dset_info[subset]["modality_names"]:
                new_entry = {}
                new_entry["Dataset"] = do.name
                new_entry["Subset"] = subset
                new_entry["Modality"] = modality
                new_entry["Num Subj"] = 0.0
                new_entry["% Processed"] = 0.0
                total_proc = 0
                for res in [64, 128, 256]:
                    label_dir = os.path.join(paths["ROOT"], "processed", f"res{res}", do.name, "label_info", subset, "all_labels.npy")
                    if os.path.exists(label_dir):
                        new_entry["Labels Known"] = True
                        label_file = np.load(label_dir)
                        new_entry["Num Subj"] = label_file[0]
                        for dt in ["maxslice", "midslice"]:
                            slice_dir = os.path.join(paths["ROOT"], "processed", f"res{res}", do.name, f"{dt}_v{version}", subset, modality, str(do.dset_info[subset]["planes"][0]))
                            if os.path.exists(slice_dir):
                                new_entry["Num Subj"] = label_file[0]
                                num_processed = len(os.listdir(slice_dir))
                                total_proc += num_processed
                                new_entry[f"{dt},{res}"] = num_processed
                            else:
                                new_entry[f"{dt},{res}"] = 0.0
                    else:
                        new_entry["Labels Known"] = False
                new_entry["% Processed"] = np.round((total_proc / (6 * new_entry["Num Subj"])) * 100, 2) if new_entry["Num Subj"] != 0 else 0.0
                dp_objects.append(new_entry)
    dataframe = pd.DataFrame(dp_objects)
    return dataframe
    
