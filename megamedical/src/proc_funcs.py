import numpy as np 
import submitit 
import math 
import matplotlib.pyplot as plt 
import os
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
import megamedical.utils as utils


def show_processing(dataset_object, subdset, version, show_hists=False):
    assert isinstance(subdset, str), "Must be a string."
    if subdset == "all":
        dset_names = list(dataset_object.dset_info.keys())
    else:
        dset_names = [subdset]
    for dataset in dset_names:
        dataset_object.proc_func(dataset, 
                                 version=version,
                                 show_hists=show_hists,
                                 show_imgs=True,
                                 save_slices=False,
                                 redo_processed=True)
      

def show_dataset(dataset_object,  
                 version, 
                 subdset="all",
                 resolution=128, 
                 collapse_labels=True,
                 subjs_to_vis=10):
    assert isinstance(subdset, str), "Must be a string."
    
    # Get list of sub-datasets
    if subdset == "all":
        dset_names = list(dataset_object.dset_info.keys())
    else:
        dset_names = [subdset]
    
    # Get list of processed dirs (if they exist)
    subdsets = dset_names
    subdsets_data_dirs = []
    for sdset in dset_names:
        try:
            sdset_info = dataset_object.dset_info[sdset]
            if version in ["megamedical_v3.1", "midslice_v3.1"]:
                dset_path = os.path.join(paths["ROOT"], "megamedical/datasets", sdset_info["main"], "processed", version, sdset)
            elif version == "v3.2-midslice":
                dset_path = os.path.join("/share/sablab/nfs02/users/gid-dalcaav/data/v3.2-midslice", sdset_info["main"], sdset)
            else:
                raise NotImplemented(f"Version of {sdset_info['main']}/{sdset} does not exist!")
            subdsets_data_dirs.append(dset_path)
        except Exception as e:
            print(e)
            continue
    # Display each of the subjects
    for subdset, sdd in zip(subdsets, subdsets_data_dirs):
        modalities = os.listdir(sdd)
        for mod in modalities:
            mod_dir = os.path.join(sdd, mod)
            subjs = os.listdir(mod_dir)
            chosen_subjs = np.random.choice(subjs, subjs_to_vis)
            if collapse_labels:
                display_collapsed(subdset, mod, mod_dir, chosen_subjs, resolution)
            else:
                display_non_collapsed(mod_dir, subjs, resolution)
                

def display_non_collapsed(mod_dir, subjs, resolution):
    for sub in subjs:
        img_dir = os.path.join(mod_dir, sub, f"img_{resolution}.npy")
        seg_dir = os.path.join(mod_dir, sub, f"seg_{resolution}.npy")
        img = np.load(img_dir)
        seg = np.load(seg_dir)
        num_planes = len(img.shape)
        for plane in range(num_planes):
            print(f"SUBDATASET: {subdset}, MODALITY: {mod}, PLANE: {plane}")
            num_images = (seg.shape[-1] + 1)
            num_cols = min(num_images, 10)
            num_rows = math.ceil(num_images/10)
            f, axarr = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=[2*num_cols,2*num_rows])
            if num_planes == 3:
                img_slice = np.take(img, img.shape[plane]//2, plane)
                seg_slice = np.take(seg, seg.shape[plane]//2, plane)
            else:
                img_slice = img
                seg_slice = seg
            if num_rows == 1:
                axarr[0].imshow(img_slice)
                axarr[0].set_xticks([])
                axarr[0].set_yticks([])
                for l_idx in range(seg.shape[-1]):
                    bin_slice = seg_slice[..., l_idx]
                    axarr[l_idx + 1].imshow(bin_slice)
                    axarr[l_idx + 1].set_xticks([])
                    axarr[l_idx + 1].set_yticks([])
            else:
                axarr[0,0].imshow(img_slice)
                axarr[0,0].set_xticks([])
                axarr[0,0].set_yticks([])
                for row in range(num_rows):
                    for col in range(num_cols):
                        index = row*num_cols + col
                        if index > 0:
                            if index < seg.shape[-1]:
                                bin_slice = seg_slice[..., index]
                                axarr[row, col].imshow(bin_slice)
                                axarr[row, col].set_xticks([])
                                axarr[row, col].set_yticks([])
                            else:
                                axarr[row, col].axis('off')
            plt.show()
            
            
def display_collapsed(subdset, mod, mod_dir, subjs, resolution): 
    images = []
    labels = []
    for sub in subjs:
        img_dir = os.path.join(mod_dir, sub, f"img_{resolution}.npy")
        seg_dir = os.path.join(mod_dir, sub, f"seg_{resolution}.npy")
        image = np.load(img_dir)
        images.append(image)
        labels.append(np.load(seg_dir))
        if len(image.shape) == 3:
            num_planes = 3
        else:
            num_planes = 1
    for plane in range(num_planes):
        print(f"SUBDATASET: {subdset}, MODALITY: {mod}, PLANE: {plane}")
        f, axarr = plt.subplots(nrows=2, ncols=len(subjs), figsize=[5*len(subjs),5])
        for idx, (img, seg) in enumerate(zip(images, labels)):
            if num_planes == 3:
                img_slice = np.take(img, img.shape[plane]//2, plane)
                seg_slice = np.take(seg, seg.shape[plane]//2, plane)
            else:
                img_slice = img
                seg_slice = seg
            if len(seg_slice.shape) > 2 and seg_slice.shape[-1] != 1:
                seg_slice = np.argmax(seg_slice, axis=-1)
            elif seg_slice.shape[-1] == 1:
                seg_slice = seg_slice[...,0]
                
            axarr[0,idx].imshow(img_slice)
            axarr[1,idx].imshow(seg_slice)
            axarr[0,idx].set_xticks([])
            axarr[0,idx].set_yticks([])
            axarr[1,idx].set_xticks([])
            axarr[1,idx].set_yticks([])
        plt.show()
                
        
def process_dataset(datasets,
                    subdsets=None,
                    save_slices=False,
                    slurm=False, 
                    visualize=False,
                    redo_processed=True,
                    show_hists=False,
                    version="4.0",
                    timeout=540):
    assert not (len(datasets) > 1 and visualize), "Can't visualize a list of processing."
    assert not (slurm and visualize), "If you are submitting slurm no vis."

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        dset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for dset in dset_names:
            if slurm:
                slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{dset}")
                executor = submitit.AutoExecutor(folder=slurm_root)
                executor.update_parameters(timeout_min=timeout, mem_gb=16,
                                           gpus_per_node=1, slurm_partition="sablab", slurm_wckey="")
                job = executor.submit(do.proc_func,
                                      dset,
                                      version,
                                      visualize,
                                      save_slices,
                                      show_hists,
                                      redo_processed)
            else:
                do.proc_func(dset,
                            pps.produce_slices,
                            version,
                            visualize,
                            save_slices,
                            show_hists,
                            redo_processed)
                    

def get_label_dist(datasets,
                    subdsets=None,
                    save_hists=False,
                    slurm=False, 
                    visualize=False,
                    version="4.0",
                    timeout=540):
    assert not (len(datasets) > 1 and visualize), "Can't visualize a list of processing."
    assert not (slurm and visualize), "If you are submitting slurm no vis."

    dataset_objects = [utils.build_dataset(ds) for ds in datasets]

    for do in dataset_objects:
        dset_names = list(do.dset_info.keys()) if subdsets is None else subdsets
        for dset in dset_names:
            if slurm:
                slurm_root = os.path.join(paths["ROOT"], f"bash/submitit/{do.name}/{dset}")
                executor = submitit.AutoExecutor(folder=slurm_root)
                executor.update_parameters(timeout_min=timeout, mem_gb=16,
                                           gpus_per_node=1, slurm_partition="sablab", slurm_wckey="")
                job = executor.submit(do.proc_func,
                                      dset,
                                      pps.label_dist,
                                      version,
                                      visualize,
                                      save_hists)
            else:
                do.proc_func(dset,
                            pps.label_dist,
                            version,
                            visualize,
                            save_hists)
