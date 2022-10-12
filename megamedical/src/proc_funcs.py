import numpy as np 
import submitit 
import math 
import matplotlib.pyplot as plt 
import shutil
import os
import pandas as pd
import glob
pd.set_option('display.max_rows',100)

# Megamedical imports
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
import megamedical.utils as utils


# Combined function of processing
def combined_pipeline_process(do,
                              subdset,
                              version,
                              resolutions,
                              parallelize,
                              redo_processed,
                              train_split,
                              save):
    print(f"Generating Unique Labels for Dataset: {do.name}, Subdset: {subdset}")
    # Gather unique labels
    pps.gather_unique_labels(data_obj=do,
                             subdset=subdset,
                             version=version,
                             resolutions=resolutions,
                             parallelize=parallelize,
                             save=save)
    print(f"Generating Pop Stats for Dataset: {do.name}, Subdset: {subdset}")
    # Get population label matrices, subj list
    pps.gather_population_statistics(data_obj=do,
                                     subdset=subdset,
                                     version=version,
                                     resolutions=resolutions,
                                     parallelize=parallelize,
                                     save=save)
    print(f"Processing Images for Dataset: {do.name}, Subdset: {subdset}")
    # Process Images
    do.proc_func(subdset=subdset,
                 pps_function=pps.produce_slices,
                 parallelize=parallelize,
                 load_images=True,
                 version=version,
                 show_imgs=False,
                 save=save,
                 show_hists=False,
                 resolutions=resolutions,
                 redo_processed=redo_processed)
    print(f"Making Training Splits for Dataset: {do.name}, Subdset: {subdset}")
    # Make training splits
    make_splits(data_obj=do,
                subdset=subdset,
                resolutions=resolutions,
                version=version,
                amount_training=train_split)
    # Finished
    print(f"Done with Dataset: {do.name}, Subdset: {subdset}!")
        
        
# High level function, can do process all at once.
def process_pipeline(datasets,
                     subdsets=None,
                     save=False,
                     slurm=False,
                     redo_processed=True,
                     resolutions=[64, 128, 256],
                     train_split=0.7,
                     version="4.0",
                     timeout=540,
                     mem_gb=32,
                     parallelize=False):
    
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        subdset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for subdset in subdset_names:
            try:
                if slurm:
                    slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{subdset}")
                    executor = submitit.AutoExecutor(folder=slurm_root)
                    executor.update_parameters(timeout_min=timeout, mem_gb=mem_gb, slurm_partition="sablab", slurm_wckey="")
                    job = executor.submit(combined_pipeline_process,
                                          do,
                                          subdset,
                                          version,
                                          resolutions,
                                          parallelize,
                                          redo_processed,
                                          train_split,
                                          save)
                else:
                    combined_pipeline_process(do,
                                              subdset,
                                              version,
                                              resolutions,
                                              parallelize, 
                                              redo_processed,
                                              train_split,
                                              save)
            except Exception as e:
                print(e)
                continue
        
        
# Middle step of processing, actually process the images:       
def process_dataset(datasets,
                    subdsets=None,
                    save=False,
                    slurm=False, 
                    show_imgs=False,
                    redo_processed=True,
                    resolutions=[64, 128, 256],
                    show_hists=False,
                    version="4.0",
                    timeout=540,
                    mem_gb=32,
                    parallelize=False):
    assert not (parallelize and show_imgs) or (parallelize and show_hists), "Parallelization disabled for showing graphics."
    assert not (slurm and show_imgs), "If you are submitting slurm no vis."
    
    load_images = True
    accumulate = False
    
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])
    
    dataset_objects = [utils.build_dataset(ds) for ds in datasets]
    
    for do in dataset_objects:
        subdset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for subdset in subdset_names:
            try:
                if slurm:
                    slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{subdset}")
                    executor = submitit.AutoExecutor(folder=slurm_root)
                    executor.update_parameters(timeout_min=timeout, mem_gb=mem_gb, slurm_partition="sablab", slurm_wckey="")
                    job = executor.submit(do.proc_func,
                                          subdset,
                                          pps.produce_slices,
                                          parallelize,
                                          load_images,
                                          accumulate,
                                          version,
                                          show_imgs,
                                          save,
                                          show_hists,
                                          resolutions,
                                          redo_processed)
                else:
                    do.proc_func(subdset,
                                 pps.produce_slices,
                                 parallelize,
                                 load_images,
                                 accumulate,
                                 version,
                                 show_imgs,
                                 save,
                                 show_hists,
                                 resolutions,
                                 redo_processed)
            except Exception as e:
                print(e)
                continue
                
                
# Middle step of processing, determine the label statistics of a dataset.
def generate_population_statistics(datasets,
                                  subdsets=None,
                                  save=False,
                                  slurm=False,
                                  resolutions=[64, 128, 256],
                                  version="4.0",
                                  timeout=180,
                                  mem_gb=16,
                                  parallelize=False):
    
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
                job = executor.submit(pps.gather_population_statistics,
                                      do,
                                      subdset,
                                      version,
                                      resolutions,
                                      parallelize,
                                      save)
            else:
                pps.gather_population_statistics(do,
                                                 subdset,
                                                 version,
                                                 resolutions,
                                                 parallelize,
                                                 save)

                
# First step of processing, determine the unique labels of a dataset.
def generate_unique_label_files(datasets,
                              subdsets=None,
                              save=False,
                              slurm=False,
                              resolutions=[64, 128, 256],
                              version="4.0",
                              redo_processed=False,
                              timeout=180,
                              mem_gb=16,
                              parallelize=False):
    
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
                job = executor.submit(pps.gather_unique_labels,
                                      do,
                                      subdset,
                                      version,
                                      resolutions,
                                      parallelize,
                                      save)
            else:
                pps.gather_unique_labels(do,
                                         subdset,
                                         version,
                                         resolutions,
                                         parallelize,
                                         save)
                    
# Table for getting current status of processing
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
                # Load the entire list of subjects
                raise NotImplementedError("Need to implement the subject file.")
                subject_file = None
                if os.path.exists(subject_file):
                    all_subs = np.load(subject_file)          
                    new_entry["Num Subj"] = len(all_subs)
                    new_entry["% Processed"] = 0.0
                    total_proc = 0
                    for res in [64, 128, 256]:
                        res_dir = os.path.join(paths["PROC"], f"res{res}", do.name)
                        label_dir = os.path.join(res_dir, "label_info", subset, "all_labels.npy")
                        new_entry["Labels Known"] = os.path.exists(label_dir)
                        for dt in ["maxslice", "midslice"]:
                            pop_paths = [os.path.exists(os.path.join(res_dir, "label_info", subset, f"{dt}_pop_lab_amount_{plane}.pickle")) for plane in do.dset_info[subset]["planes"]]
                            new_entry[f"{res} {dt} Pop Known"] = np.all(pop_paths)
                            slice_dir = os.path.join(res_dir, f"{dt}_v{version}", subset, modality, str(do.dset_info[subset]["planes"][0]))
                            if os.path.exists(slice_dir):
                                num_processed = len(os.listdir(slice_dir))
                                total_proc += num_processed
                                new_entry[f"{dt},{res}"] = num_processed
                            else:
                                new_entry[f"{dt},{res}"] = 0.0
                    new_entry["% Processed"] = np.round((total_proc / (6 * new_entry["Num Subj"])) * 100, 2) if new_entry["Num Subj"] != 0 else 0.0
                else:
                    new_entry["Num Subj"] = None
                    new_entry["% Processed"] = 0.0
                    new_entry["Labels Known"] = False
                dp_objects.append(new_entry)
    dataframe = pd.DataFrame(dp_objects)
    return dataframe


# Function at the end of processing to produce splits for a dataset
def make_splits(data_obj,
                subdset,
                resolutions=[64, 128, 256],
                version="4.0",
                amount_training=0.7):
    
    split_files_dir = os.path.join(paths["PROC"], "split_files")
    if not os.path.exists(split_files_dir):
        os.makedirs(split_files_dir)
    
    for res in resolutions:
        for dset_type in ["midslice_v4.0", "maxslice_v4.0"]:
            try:
                subjects = np.array(utils.proc_utils.get_list_of_subjects(paths["PROC"], res, dset_type, data_obj.name, subdset))

                total_amount = len(subjects)
                indices = np.arange(total_amount)
                np.random.shuffle(indices)

                train_amount = int(total_amount*amount_training)
                val_test_amount = total_amount - train_amount
                val_amount = int(val_test_amount*0.5)
                test_amount = val_test_amount - val_amount

                train_indices = indices[:train_amount]
                val_indices = indices[train_amount:train_amount+val_amount]
                test_indices = indices[-test_amount:]

                names_dict = {
                    "train": subjects[train_indices],
                    "val": subjects[val_indices], 
                    "test": subjects[test_indices]
                }

                for split in ["train", "val", "test"]:
                    split_file = open(os.path.join(split_files_dir, f"res{res}__{data_obj.name}__{dset_type}__{subdset}__{split}.txt"), "w")
                    for file_name in names_dict[split]:
                        split_file.write(file_name + "\n")
                    split_file.close()

            except Exception as e:
                print("Error:", e)

                
# get rid of processed datasets
def flush_processed_datasets(datasets):
    
    if datasets == "all":
        confirmation = input("Are you SURE. Y/y:")
        if confirmation in ["y", "Y"]:   
            datasets = os.listdir(paths["DATA"])
        else:
            return 0 
    
    dataset_objects = [utils.build_dataset(ds) for ds in datasets]
    
    for do in dataset_objects:
        for res in ["res64", "res128", "res256"]:
            res_dir = os.path.join(paths["PROC"], res, do.name)
            if os.path.exists(res_dir):
                shutil.rmtree(res_dir)
    
            split_files = glob.glob(os.path.join(paths["PROC"], "split_files", f"{res}__{do.name}*"))
            for sf in split_files:
                os.remove(sf)
    



