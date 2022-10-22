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
                   save_name,
                   loaded_image,
                   loaded_label,
                   dset_info,
                   resolutions,
                   save,
                   show_hists,
                   show_imgs):
    
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
        
        for res in resolutions:
            # Get all of the labels in the volume population, note that the first index tracks the number
            # of subjects.
            unique_labels = np.load(os.path.join(root_dir, f"res{res}", dset_info["main"], "label_info", subdset, "all_labels.npy"))

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
                bin_mask = np.float32(square_label==label)

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
                save_maxslice(max_save_dir, image_res, seg_res, subdset, mode, save_name, dset_info["planes"], max_slices)
                save_midslice(mid_save_dir, image_res, seg_res, subdset, mode, save_name, dset_info["planes"])
                
                
# Produce the "all label" matrices for each resolution
# and per label statistics for an entire dataset.
def gather_population_statistics(data_obj,
                                 subdset,
                                 version,
                                 resolutions,
                                 parallelize,
                                 redo_processed,
                                 save):
      
    # total_label_info is a dictionary that is structed like the following
    # total_label_info
    # - resolution (64, 128, 256, etc.)
    #     - label amounts midslice (per plane)
    #         - plane 0,1,2...
    #     - label amounts maxslice (per plane)
    #         - plane 0,1,2...
    data_obj.proc_func(subdset=subdset,
                       task="stats",
                       pps_function=get_label_amounts,
                       parallelize=parallelize,
                       load_images=False,
                       accumulate=False,
                       version=version,
                       show_imgs=False,
                       save=save,
                       show_hists=False,
                       resolutions=resolutions,
                       redo_processed=redo_processed)
    
    
    for res in resolutions:
        subdset_res_root = os.path.join(paths["PROC"], f"res{res}", data_obj.name, "label_info", subdset)
        
        subj_label_file_dir = os.path.join(subdset_res_root, "pop_info_files")
        res_processed_subjs = [subj_name.replace('.pickle', '') for subj_name in os.listdir(subj_label_file_dir)]
        res_label_info = [load_obj(os.path.join(subj_label_file_dir, subj_name)) for subj_name in os.listdir(subj_label_file_dir)]
        num_subjects = len(res_label_info)
        midslice_label_info = [li["midslice"] for li in res_label_info]
        maxslice_label_info = [li["maxslice"] for li in res_label_info]
        
        # get all possible labels
        unique_labels = np.load(os.path.join(subdset_res_root, "all_labels.npy")).tolist()

        # define an inverse map going from labels to indices in the list
        label_map = {lab: unique_labels.index(lab) for lab in unique_labels}

        slice_info_set = {
            "midslice": midslice_label_info, 
            "maxslice": maxslice_label_info
        }
        
        num_unique_labels = len(unique_labels)
        for slice_type in slice_info_set.keys():
            
            for plane in data_obj.dset_info[subdset]["planes"]:
                
                label_info_array = np.zeros((num_subjects, num_unique_labels))
                
                for subj_idx, slice_info in enumerate(slice_info_set[slice_type]):
                    
                    for l_idx in range(num_unique_labels):
                                           
                        label_info_array[subj_idx, l_idx] = slice_info[plane][l_idx]
                                           
                if save:
                    dict_dir = os.path.join(subdset_res_root, f"{slice_type}_pop_lab_amount_{plane}")
                    dict_pair = {
                        "index": res_processed_subjs,
                        "pop_label_amount": label_info_array
                    }
                    dump_dictionary(dict_pair, dict_dir)


# Produce the "all label" matrices for each resolution
# and per label statistics for an entire dataset.
def gather_unique_labels(data_obj,
                         subdset,
                         version,
                         resolutions,
                         parallelize,
                         redo_processed,
                         save):
    save_dirs = [os.path.join(os.path.join(paths['ROOT'], "processed"), f"res{res}", data_obj.name, "label_info", subdset, "all_labels.npy") for res in resolutions]
    if (not redo_processed) and np.all([os.path.exists(save_dir) for save_dir in save_dirs]):
        return
        
    proc_dir, processed_subjects, resolution_label_dict = data_obj.proc_func(subdset=subdset,
                                                                             task="labels",
                                                                             pps_function=get_all_unique_labels,
                                                                             parallelize=parallelize,
                                                                             load_images=False,
                                                                             accumulate=True,
                                                                             version=version,
                                                                             show_imgs=False,
                                                                             save=save,
                                                                             show_hists=False,
                                                                             resolutions=resolutions,
                                                                             redo_processed=True
                                                                             )                 
    for res in resolutions:
        total_label_info = resolution_label_dict[res]
        num_subjects = len(total_label_info)
        unique_labels = sorted(list(set([label for subj in total_label_info for label in subj])))
        if save and len(unique_labels) > 0:
            save_dir = os.path.join(os.path.join(paths['ROOT'], "processed"), f"res{res}", data_obj.name, "label_info", subdset)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            all_label_dir = os.path.join(save_dir, "all_labels.npy")
            np.save(all_label_dir, np.array(unique_labels))