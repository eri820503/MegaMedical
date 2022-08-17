# random imports
import numpy as np
import os

from megamedical.utils.registry import paths
from megamedical.utils.proc_utils import *

def make_processed_dir(dset, subdset, save_slices, dataset_ver):
    root_dir = os.path.join(paths["DATA"], dset, f"processed")
    
    dirs_to_make = []
    for prefix in ["megamedical", "maxslice", "midslice"]:
        dirs_to_make.append(os.path.join(root_dir, f"{prefix}_v{dataset_ver}", subdset))

    if save_slices:
        for dtm in dirs_to_make:
            if not os.path.exists(dtm):
                os.makedirs(dtm)
        for mode in dset_info["modality_names"]:
            for subset in dirs_to_make:
                mode_dir = os.path.join(subset, mode)
                if not os.path.exists(mode_dir):
                    os.makedirs(mode_dir)
    return root_dir


def save_megamedical(root_dir,
                     dataset_ver, 
                     mode,
                     subject_name,
                     image, 
                     seg, 
                     resolution):
    proc_stems = ["megamedical", "maxslice", "midslice"]
    for ps in proc_stems:
        proc_dir = os.path.join(root_dir, f"{ps}_v{dataset_ver}", mode, subject_name)
        if ps == "megamedical":
            np.save(os.path.join(proc_dir, f"img_{resolution}.npy"), image)
            np.save(os.path.join(proc_dir, f"seg_{resolution}.npy"), seg)
        elif ps == "maxslice":
            np.save(os.path.join(proc_dir, f"img_{resolution}.npy"), image)
            np.save(os.path.join(proc_dir, f"seg_{resolution}.npy"), seg)
        else:
            np.save(os.path.join(proc_dir, f"img_{resolution}.npy"), image)
            np.save(os.path.join(proc_dir, f"seg_{resolution}.npy"), seg)


def produce_slices(root_dir,
                   version,
                   dataset,
                   subject_name,
                   loaded_image,
                   loaded_label,
                   dset_info,
                   resolutions=[256,128],
                   save_slices=False, 
                   show_hists=False,
                   show_imgs=False):
    # Set the name to be saved
    save_name = subject_name.split(".")[0]
    
    #get all of the labels in the volume, once 
    unique_labels = np.unique(loaded_label)[1:]
    
    for idx, mode in enumerate(dset_info["modality_names"]):
        #Extract the modality if it exists (mostly used for MSD)
        if len(dset_info["modality_names"]) != 1:
            modality_loaded_image = loaded_image[:,:,:,idx]
        else:
            modality_loaded_image = loaded_image
       
        if dset_info["do_clip"]:
            #clip volume between prespecified values
            modality_loaded_image = clip_volume(modality_loaded_image, 
                                                dset_info["norm_scheme"], 
                                                dset_info["clip_args"])
        
        #normalize the volume between [0,1]
        normalized_modality_image = relative_norm(modality_loaded_image)
        
        if show_hists:
            display_histogram(normalized_modality_image.flatten())
        
        #make the volume/label a 3D cube
        square_image = squarify(normalized_modality_image)
        square_label = squarify(loaded_label)
        
        #original square image size
        old_size = square_image.shape[0]
        
        for res in resolutions:
            #Resize to several resolutions
            image_res = blur_and_resize(square_image, old_size, new_size=res, order=1)

            if show_imgs:
                for plane in dset_info["planes"]:
                    display_processing_slices(square_image, square_label, plane)

            #final segmentations are with labels in the last dimension
            if len(square_image.shape) == 2:
                seg_res = np.zeros((res, res, len(unique_labels)))
            else:     
                seg_res = np.zeros((res, res, res, len(unique_labels)))

            #go through unique labels and add to slices
            for lab_idx, label in enumerate(unique_labels):
                #isolate mask of label
                label = int(label)
                bin_mask = np.float32((square_label==label))

                #produce resized segmentations
                bin_seg_res = blur_and_resize(bin_mask, old_size, new_size=res, order=0)

                #place resized segs in regular arrays
                seg_res[..., lab_idx] = bin_seg_res

            if save_slices:
                save_megamedical(root_dir, 
                                 version, 
                                 mode,
                                 subject_name,
                                 image_res, 
                                 seg_res, 
                                 res)