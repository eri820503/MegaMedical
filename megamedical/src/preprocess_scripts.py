# random imports
import numpy as np
import os

from megamedical.utils.registry import paths
from megamedical.utils.proc_utils import *

def make_processed_dir(dset, subdset, save_slices, dataset_ver):
    root_dir = os.path.join(paths["DATA"], dset, f"processed")
    
    megamedical_root = os.path.join(root_dir, f"megamedical_v{dataset_ver}")
    megamedical_proc = os.path.join(megamedical_root, subdset)
    
    maxslice_root = os.path.join(root_dir, f"maxslice_v{dataset_ver}")
    maxslice_proc = os.path.join(maxslice_root, subdset)
    
    midslice_root = os.path.join(root_dir, f"midslice_v{dataset_ver}")
    midslice_proc = os.path.join(midslice_root, subdset)
    
    print("Starting:", megamedical_proc)

    if save_slices:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if not os.path.exists(megamedical_root):
            os.makedirs(megamedical_root)
        if not os.path.exists(megamedical_proc):
            os.makedirs(megamedical_proc)
        if not os.path.exists(maxslice_root):
            os.makedirs(maxslice_root)
        if not os.path.exists(maxslice_proc):
            os.makedirs(maxslice_proc)
        if not os.path.exists(midslice_root):
            os.makedirs(midslice_root)
        if not os.path.exists(midslice_proc):
            os.makedirs(midslice_proc)
            
        for mode in dset_info["modality_names"]:
            mode_megamedical = os.path.join(megamedical_proc, mode)
            mode_maxslice = os.path.join(maxslice_proc, mode)
            mode_midslice = os.path.join(midslice_proc, mode)
            if not os.path.exists(mode_megamedical):
                os.makedirs(mode_megamedical)
            if not os.path.exists(mode_maxslice):
                os.makedirs(mode_maxslice)
            if not os.path.exists(mode_midslice):
                os.makedirs(mode_midslice)
    
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