import numpy as np
import submitit
import math
import matplotlib.pyplot as plt
import shutil
import os
import pandas as pd
import glob
import tarfile
pd.set_option('display.max_rows',100)

# Megamedical imports
from megamedical.src import preprocess_scripts as pps
from megamedical.src import checks as check
from megamedical.src import thunder
from megamedical.utils.registry import paths
import megamedical.utils as utils


# Combined function of processing ["labels","stats","images","splits"]
def combined_pipeline_process(steps,
                              do,
                              subdset,
                              version,
                              resolutions,
                              parallelize,
                              redo_processed,
                              train_split,
                              save):
    if "labels" in steps:
        print(f"Generating Unique Labels for Dataset: {do.name}, Subdset: {subdset}")
        # Gather unique labels
        pps.gather_unique_labels(data_obj=do,
                                 subdset=subdset,
                                 version=version,
                                 resolutions=resolutions,
                                 parallelize=parallelize,
                                 redo_processed=redo_processed,
                                 save=save)

    if "stats" in steps:
        print(f"Generating Pop Stats for Dataset: {do.name}, Subdset: {subdset}")
        # Get population label matrices, subj list
        pps.gather_population_statistics(data_obj=do,
                                         subdset=subdset,
                                         version=version,
                                         resolutions=resolutions,
                                         parallelize=parallelize,
                                         redo_processed=redo_processed,
                                         save=save)

    if "images" in steps:
        print(f"Processing Images for Dataset: {do.name}, Subdset: {subdset}")
        # Process Images
        do.proc_func(subdset=subdset,
                     task="images",
                     pps_function=pps.produce_slices,
                     parallelize=parallelize,
                     load_images=True,
                     version=version,
                     show_imgs=False,
                     save=save,
                     show_hists=False,
                     resolutions=resolutions,
                     redo_processed=redo_processed)

    if "splits" in steps:
        print(f"Making Training Splits for Dataset: {do.name}, Subdset: {subdset}")
        # Make training splits
        make_splits(data_obj=do,
                    subdset=subdset,
                    resolutions=resolutions,
                    version=version,
                    amount_training=train_split,
                    save=save)
    # Finished
    print(f"Done with Dataset: {do.name}, Subdset: {subdset}!")


# High level function, can do process all at once.
def process_pipeline(steps,
                     datasets,
                     ignore_datasets=None,
                     ignore_subdsets=None,
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

    # Avoid double processing
    if ignore_datasets:
        for dset in ignore_datasets:
            datasets.remove(dset)

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        subdset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for subdset in subdset_names:
            if ignore_subdsets is None or subdset not in ignore_subdsets:
                if slurm:
                    slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{subdset}")
                    # Reset the directory if it exists
                    if os.path.exists(slurm_root):
                        shutil.rmtree(slurm_root)
                    executor = submitit.AutoExecutor(folder=slurm_root)
                    executor.update_parameters(timeout_min=timeout, mem_gb=mem_gb, slurm_partition="sablab", slurm_wckey="")
                    job = executor.submit(combined_pipeline_process,
                                          steps,
                                          do,
                                          subdset,
                                          version,
                                          resolutions,
                                          parallelize,
                                          redo_processed,
                                          train_split,
                                          save)
                else:
                    combined_pipeline_process(steps,
                                              do,
                                              subdset,
                                              version,
                                              resolutions,
                                              parallelize,
                                              redo_processed,
                                              train_split,
                                              save)


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
            if slurm:
                slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{subdset}")
                # Reset the directory if it exists
                if os.path.exists(slurm_root):
                    shutil.rmtree(slurm_root)
                executor = submitit.AutoExecutor(folder=slurm_root)
                executor.update_parameters(timeout_min=timeout, mem_gb=mem_gb, slurm_partition="sablab", slurm_wckey="")
                task = "images"
                job = executor.submit(do.proc_func,
                                      subdset,
                                      task,
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
                task = "images"
                do.proc_func(subdset,
                             task,
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


# Middle step of processing, determine the label statistics of a dataset.
def generate_population_statistics(datasets,
                                  subdsets=None,
                                  save=False,
                                  slurm=False,
                                  resolutions=[64, 128, 256],
                                  version="4.0",
                                  timeout=180,
                                  mem_gb=16,
                                  redo_processed=True,
                                  parallelize=False):

    if datasets == "all":
        datasets = os.listdir(paths["DATA"])

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        subdset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for subdset in subdset_names:
            if slurm:
                slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{subdset}")
                # Reset the directory if it exists
                if os.path.exists(slurm_root):
                    shutil.rmtree(slurm_root)
                executor = submitit.AutoExecutor(folder=slurm_root)
                executor.update_parameters(timeout_min=timeout, mem_gb=mem_gb, slurm_partition="sablab", slurm_wckey="")
                job = executor.submit(pps.gather_population_statistics,
                                      do,
                                      subdset,
                                      version,
                                      resolutions,
                                      parallelize,
                                      redo_processed,
                                      save)
            else:
                pps.gather_population_statistics(do,
                                                 subdset,
                                                 version,
                                                 resolutions,
                                                 parallelize,
                                                 redo_processed,
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
                # Reset the directory if it exists
                if os.path.exists(slurm_root):
                    shutil.rmtree(slurm_root)
                executor = submitit.AutoExecutor(folder=slurm_root)
                executor.update_parameters(timeout_min=timeout, mem_gb=mem_gb, slurm_partition="sablab", slurm_wckey="")
                job = executor.submit(pps.gather_unique_labels,
                                      do,
                                      subdset,
                                      version,
                                      resolutions,
                                      parallelize,
                                      redo_processed,
                                      save)
            else:
                pps.gather_unique_labels(do,
                                         subdset,
                                         version,
                                         resolutions,
                                         parallelize,
                                         redo_processed,
                                         save)

# Table for getting current status of processing
def get_processing_status(datasets,
                          hide_disfunctional=True,
                          version="4.0"):

    if datasets == "all":
        datasets = os.listdir(paths["DATA"])

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    dp_objects = []
    for do in dataset_objects:
        for subset in do.dset_info.keys():
            if not hide_disfunctional or (not "functional" in do.dset_info[subset].keys()):
                new_entry = {}
                new_entry["Dataset"] = do.name
                new_entry["Subset"] = subset
                new_entry["Modalities"] = do.dset_info[subset]["modality_names"]
                new_entry["Planes"] = do.dset_info[subset]["planes"]
                label_files = [os.path.join(paths["PROC"],f"res{res}", do.name, "label_info", subset, "all_labels.npy") for res in [64, 128, 256]]

                # Unique label files
                if np.all([os.path.exists(label_file) for label_file in label_files]):
                    # Load 64 version as they should all be the same.
                    new_entry["Labels"] = np.load(label_files[0])
                else:
                    new_entry["Labels"] = None

                # Population statistics
                proc_data_files = []
                for res in [64, 128, 256]:
                    for plane in do.dset_info[subset]["planes"]:
                        proc_data_files.append(os.path.join(paths["PROC"],f"res{res}", do.name, "label_info", subset, f"midslice_pop_lab_amount_{plane}.pickle"))
                new_entry["Pop Stats Known"] = np.all([os.path.exists(pdf) for pdf in proc_data_files])

                if new_entry["Pop Stats Known"]:
                    stats_file = utils.proc_utils.load_obj(proc_data_files[0])
                    new_entry["Num Subjs"] = len(stats_file["index"])
                    proc_dir_64 = os.path.join(paths["PROC"], "res64", do.name, f"midslice_v{version}", subset, new_entry["Modalities"][0], str(new_entry["Planes"][0]))
                    proc_dir_128 = proc_dir_64.replace("res64", "res128")
                    proc_dir_256 = proc_dir_64.replace("res64", "res256")
                    if os.path.exists(proc_dir_64) and new_entry["Num Subjs"] != 0:
                        new_entry["% 64 Proc"] = np.round(len(os.listdir(proc_dir_64))/new_entry["Num Subjs"], 3) * 100
                    else:
                        new_entry["% 64 Proc"] = None
                    if os.path.exists(proc_dir_128) and new_entry["Num Subjs"] != 0:
                        new_entry["% 128 Proc"] = np.round(len(os.listdir(proc_dir_128))/new_entry["Num Subjs"], 3) * 100
                    else:
                        new_entry["% 128 Proc"] = None
                    if os.path.exists(proc_dir_256) and new_entry["Num Subjs"] != 0:
                        new_entry["% 256 Proc"] = np.round(len(os.listdir(proc_dir_256))/new_entry["Num Subjs"], 3) * 100
                    else:
                        new_entry["% 256 Proc"] = None
                else:
                    new_entry["Num Subjs"] = None
                    new_entry["% 64 Proc"] = None
                    new_entry["% 128 Proc"] = None
                    new_entry["% 256 Proc"] = None
                dp_objects.append(new_entry)
    dataframe = pd.DataFrame(dp_objects)
    return dataframe


# Function at the end of processing to produce splits for a dataset
def make_splits(data_obj,
                subdset,
                resolutions=[64, 128, 256],
                version="4.0",
                amount_training=0.7,
                save=False):

    proc_raw_root = os.path.join(paths["PROC"], "processed_raw")
    split_files_dir = os.path.join(proc_raw_root, "split_files")
    if not os.path.exists(split_files_dir):
        os.makedirs(split_files_dir)

    res_proc_subjs = []
    for res in resolutions:
        for dset_type in ["midslice_v4.0", "maxslice_v4.0"]:
            res_proc_subjs.append(len(utils.proc_utils.get_list_of_subjects(proc_raw_root, 64, "midslice_v4.0", data_obj.name, subdset)))

    res_num_subjs = (np.array(res_proc_subjs) == res_proc_subjs[0])
    assert np.all(res_num_subjs), "Make sure all things are equivalently processed."

    # Default to res64 and dset type midslice_v4.0
    subjects = np.array(utils.proc_utils.get_list_of_subjects(proc_raw_root, 64, "midslice_v4.0", data_obj.name, subdset))

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

    if save:
        for split in ["train", "val", "test"]:
            split_file = open(os.path.join(split_files_dir, f"{data_obj.name}__{subdset}__{split}.txt"), "w")
            for file_name in names_dict[split]:
                split_file.write(file_name + "\n")
            split_file.close()


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


# get rid of processed datasets
def replace_label_file(dataset,
                       subset,
                       new_labels):
    for res in ["res64", "res128", "res256"]:
        label_file_dir = os.path.join(paths["PROC"], res, dataset, "label_info", subset, "all_labels.npy")
        np.save(label_file_dir, np.array(new_labels))


# get rid of processed datasets
def check_datasets(datasets,
                   subdsets=None):

    if datasets == "all":
        datasets = os.listdir(paths["DATA"])

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        subdset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for subdset in subdset_names:
            check.verify_dataset(do, subdset)


# get rid of processed datasets
def thunderify_datasets(datasets,
                        move_datasets,
                        export_name,
                        force=True,
                        max_workers=16,
                        suppress_exceptions=False,
                        dry_run=False,
                        version=4.1,
                        clean_dirs=True):
    assert export_name != "", "Must have non-empty export name."

    if move_datasets:
        root = "/home/vib9/src/MegaMedical/processed"
        for res in ["res64", "res128", "res256"]:
            for dset in datasets:
                old_dir = os.path.join(root, res, dset)
                new_dir_root = os.path.join(root, "processed_raw", res)
                if not os.path.exists(new_dir_root):
                    os.makedirs(new_dir_root)
                new_dir = os.path.join(new_dir_root, dset)
                shutil.move(old_dir, new_dir)

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        subdset_names = list(do.dset_info.keys())
        for subdset in subdset_names:
            make_splits(do,
                        subdset,
                        resolutions=[64, 128, 256],
                        version="4.0",
                        amount_training=0.7,
                        save=True)

    thunder.reprocess_hierarchy(force,
                                max_workers,
                                suppress_exceptions,
                                dry_run,
                                version)

    source_dir = "/home/vib9/src/MegaMedical/processed/lmdb_processed/4.1"
    output_filename = f"/home/vib9/src/MegaMedical/processed/lmdb_processed/{export_name}"
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
