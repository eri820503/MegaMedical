# random imports
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from megamedical.utils.registry import paths
from megamedical.utils.proc_utils import *

    
def produce_slices(root_dir,
                   version,
                   subdset,
                   subject_name,
                   loaded_image,
                   loaded_label,
                   dset_info,
                   res,
                   save=False,
                   show_hists=False,
                   show_imgs=False):
    # Set the name to be saved
    save_name = subject_name.split(".")[0]
    
    for idx, mode in enumerate(dset_info["modality_names"]):
        #Extract the modality if it exists (mostly used for MSD)
        if len(dset_info["modality_names"]) != 1:
            modality_loaded_image = loaded_image[:,:,:,idx]
        else:
            modality_loaded_image = loaded_image
        
        if show_hists:
            display_histogram(modality_loaded_image.flatten())
        
        #clip volume between prespecified values
        modality_loaded_image = clip_volume(modality_loaded_image, 
                                            dset_info["norm_scheme"], 
                                            dset_info["clip_args"])
        
        if show_hists:
            display_histogram(modality_loaded_image.flatten())
        
        #normalize the volume between [0,1]
        normalized_modality_image = relative_norm(modality_loaded_image)
        
        #make the volume/label a 3D cube
        square_image = squarify(normalized_modality_image)
        square_label = squarify(loaded_label)
        
        #original square image size
        old_image_size = square_image.shape[0]
        old_seg_size = square_label.shape[0]
        
        #show midslices
        if show_imgs:
            for plane in dset_info["planes"]:
                display_processing_slices(square_image, square_label, plane)

        # Get all of the labels in the volume population, note that the first index tracks the number
        # of subjects.
        unique_labels = np.load(os.path.join(root_dir, f"res{res}", dset_info["main"], "label_info", subdset, "all_labels.npy"))
        unique_labels = np.delete(unique_labels, 0)

        #Resize to several resolutions
        image_res = blur_and_resize(square_image, old_image_size, new_size=res, order=1)

        #final segmentations are with labels in the last dimension
        if len(square_image.shape) == 2:
            seg_res = np.zeros((res, res, len(unique_labels)))
        else:     
            seg_res = np.zeros((res, res, res, len(unique_labels)))

        #go through unique labels and add to slices
        max_slices = {pl : np.zeros((len(unique_labels))) for pl in dset_info["planes"]}

        for lab_idx, label in enumerate(unique_labels):
            #isolate mask of label
            label = int(label)
            bin_mask = np.float32((square_label==label))

            #produce resized segmentations
            bin_seg_res = blur_and_resize(bin_mask, old_seg_size, new_size=res, order=0)

            # Gather maxslice info
            if len(bin_seg_res.shape) == 3:
                for pl in dset_info["planes"]:
                    all_axes = [0, 1, 2]
                    all_axes.remove(pl)
                    greatest_index = np.argmax(np.count_nonzero(bin_seg_res, axis=tuple(all_axes)))
                    max_slices[pl][lab_idx] = greatest_index

            # Place resized segs in regular array
            seg_res[..., lab_idx] = bin_seg_res

        if save:
            #Save file directories
            max_save_dir = os.path.join(root_dir, f"res{res}", dset_info["main"], f"maxslice_v{version}")
            mid_save_dir = os.path.join(root_dir, f"res{res}", dset_info["main"], f"midslice_v{version}")
            #Save each type of file
            save_maxslice(max_save_dir, image_res, seg_res, subdset, mode, subject_name, dset_info["planes"], max_slices)
            save_midslice(mid_save_dir, image_res, seg_res, subdset, mode, subject_name, dset_info["planes"])
                
                
# Produce the "all label" matrices for each resolution
# and per label statistics for an entire dataset.
def label_info(data_obj,
               subdset,
               version,
               resolutions,
               save):
    
    # total_label_info is a dictionary that is structed like the following
    # total_label_info
    # - resolution (64, 128, 256, etc.)
    #     - all labels (after resize)
    #     - label amounts midslice (per plane)
    #         - plane 0,1,2...
    #     - label amounts maxslice (per plane)
    #         - plane 0,1,2...
    proc_dir, total_label_info = data_obj.proc_func(subdset,
                                                 get_label_amounts,
                                                 load_images=False,
                                                 accumulate=True,
                                                 version=version,
                                                 resolutions=resolutions,
                                                 save=save)
    print(total_label_info.keys())
    for res in resolutions:
        res_label_info = total_label_info[res]
        num_subjects = len(res_label_info)
        print(type(res_label_info))
        total_label_info = [li["all_labels"] for li in res_label_info]
        midslice_label_info = [li["midslice"] for li in res_label_info]
        maxslice_label_info = [li["maxslice"] for li in res_label_info]

        unique_labels = sorted(list(set([label for subj in total_label_info for label in subj])))

        # define an inverse map going from labels to indices in the list
        label_map = {lab: unique_labels.index(lab) for lab in unique_labels}

        slice_info_set = {
            "midslice": midslice_label_info, 
            "maxslice": maxslice_label_info
        }

        save_dir = os.path.join(proc_dir, f"res{res}", data_obj.name, "label_info", subdset)

        if save and len(unique_labels) > 0:
            # Keep track of number of subjects, its useful (but also causes problems >:( )
            unique_labels.insert(0, num_subjects)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            all_label_dir = os.path.join(save_dir, f"all_labels.npy")
            np.save(all_label_dir, np.array(unique_labels))

        for slice_type in slice_info_set.keys():
            for plane in data_obj.dset_info[subdset]["planes"]:
                label_info_array = np.zeros((len(total_label_info), len(unique_labels)))
                for subj_idx, slice_info in enumerate(slice_info_set[slice_type]):
                    for label in slice_info[plane].keys():
                        label_info_array[subj_idx, label_map[int(label)]] = slice_info[plane][label]
                if save:
                    dict_dir = os.path.join(save_dir, f"{slice_type}_pop_lab_amount_{plane}")
                    np.save(dict_dir, label_info_array)