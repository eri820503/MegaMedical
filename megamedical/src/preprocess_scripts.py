# random imports
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from megamedical.utils.registry import paths
from megamedical.utils.proc_utils import *


def save_maxslice(proc_dir, image_res, seg_res, subdset, mode, subject_name, planes, maxslices):
        
    for plane in planes:     
        max_subject_dir = os.path.join(proc_dir, subdset, mode, str(plane), subject_name) 
        
        if not os.path.exists(max_subject_dir):
            os.makedirs(max_subject_dir)
        
        maxslice_img_dir = os.path.join(max_subject_dir, f"img.npy")
        maxslice_seg_dir = os.path.join(max_subject_dir, f"seg.npy")
 
        # Get midslice slices
        subj_shape = image_res.shape
        if len(subj_shape) == 2:
            maxslice_img = np.repeat(image_res[..., np.newaxis], seg_res.shape[-1], axis=2)
            maxslice_seg = seg_res
        else:
            slices = maxslices[plane]
            maxslice_img = np.stack([np.take(image_res, int(si), plane) for si in slices], axis=2)
            maxslice_seg = []
            for s_idx, si in enumerate(slices):
                maxslice_seg.append(np.take(seg_res, int(si), plane)[...,s_idx:s_idx+1])
            maxslice_seg = np.concatenate(maxslice_seg, axis=2)
        
        np.save(maxslice_img_dir, maxslice_img)
        np.save(maxslice_seg_dir, maxslice_seg)


def save_midslice(proc_dir, image_res, seg_res, subdset, mode, subject_name, planes):
    
    for plane in planes:   
        
        mid_subject_dir = os.path.join(proc_dir, subdset, mode, str(plane), subject_name) 
        
        if not os.path.exists(mid_subject_dir):
            os.makedirs(mid_subject_dir)
            
        midslice_img_dir = os.path.join(mid_subject_dir, f"img.npy")
        midslice_seg_dir = os.path.join(mid_subject_dir, f"seg.npy")
        
        # Get midslice slices
        subj_shape = image_res.shape
        if len(subj_shape) == 2:
            midslice_img = image_res
            midslice_seg = seg_res
        else:
            midslice_img = np.take(image_res, image_res.shape[plane]//2, plane)
            midslice_seg = np.take(seg_res, seg_res.shape[plane]//2, plane)
        
        np.save(midslice_img_dir, midslice_img)
        np.save(midslice_seg_dir, midslice_seg)

    
def produce_slices(root_dir,
                   version,
                   subdset,
                   subject_name,
                   loaded_image,
                   loaded_label,
                   dset_info,
                   resolutions,
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
        
        if dset_info["do_clip"]:
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
        old_size = square_image.shape[0]
        
        #show midslices
        if show_imgs:
            for plane in dset_info["planes"]:
                display_processing_slices(square_image, square_label, plane)
        
        for res in resolutions:
            # Get all of the labels in the volume population, note that the first index tracks the number
            # of subjects.
            unique_labels = np.load(os.path.join(root_dir, f"res{res}", dset_info["main"], "label_info", subdset, "all_labels.npy"))
            unique_labels = np.delete(unique_labels, 0)
            
            #Resize to several resolutions
            image_res = blur_and_resize(square_image, old_size, new_size=res, order=1)

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
                bin_seg_res = blur_and_resize(bin_mask, old_size, new_size=res, order=0)
                
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
                
                
def label_info(data_obj,
               subdset,
               version,
               resolutions,
               save):
    
    proc_dir, label_info = data_obj.proc_func(subdset,
                                             get_label_amounts,
                                             load_images=False,
                                             accumulate=True,
                                             version=version,
                                             resolutions=resolutions,
                                             save=save)
    
    total_label_info = [li[0] for li in label_info]
    midslice_label_info = [li[1] for li in label_info]
    maxslice_label_info = [li[2] for li in label_info]
    
    num_subjects = len(label_info)
    unique_labels = sorted(list(set([label for subj in total_label_info for label in subj])))
    
    # define an inverse map going from labels to indices in the list
    label_map = {lab: unique_labels.index(lab) for lab in unique_labels}
    
    slice_info_set = {
        "midslice": midslice_label_info, 
        "maxslice": maxslice_label_info
    }
    
    save_dir = os.path.join(proc_dir, "label_info", subdset)
    
    if save and len(unique_labels) > 0:
        # Keep track of number of subjects, its useful (but also causes problems >:( )
        unique_labels.insert(0, num_subjects)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        all_label_dir = os.path.join(save_dir, f"all_labels")
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


def get_label_amounts(proc_dir,
                      version,
                      dset_name,
                      image, 
                      loaded_image,
                      loaded_label,
                      dset_info,
                      show_hists,
                      show_imgs,
                      resolutions,
                      save):
    planes = dset_info["planes"]
    
    
    all_labels = np.unique(loaded_label)
    all_labels = np.delete(all_labels, [0])
    
    res_dict = {}
    for res in resolutions:
        res_dict[res] = {}
        maxslice_amount_dict = {}
        midslice_amount_dict = {}
        
        lab_shape = loaded_label.shape
        if len(lab_shape) == 2:
            midslice_amount_dict[0] = {lab : np.count_nonzero((loaded_label==lab).astype(int)) for lab in all_labels}
            maxslice_amount_dict[0] = {lab : np.count_nonzero((loaded_label==lab).astype(int)) for lab in all_labels}
        else:
            for plane in planes:
                all_axes = [0,1,2]
                all_axes.remove(plane)

                midslice = np.take(loaded_label, lab_shape[plane]//2, plane)
                mid_unique_labels = np.unique(midslice)
                # Get rid of 0 as a unique label
                midslice_plane_labels = np.delete(mid_unique_labels, [0])

                midslice_amount_dict[plane] = {lab : np.count_nonzero((midslice==lab).astype(int)) for lab in midslice_plane_labels}
                maxslice_amount_dict[plane] = {lab : np.amax(np.count_nonzero((loaded_label==lab).astype(int), axis=tuple(all_axes))) for lab in all_labels}
        
        res_dict[res]["midslice"] = midslice_amount_dict
        res_dict[res]["maxslice"] = maxslice_amount_dict

    return all_labels, res_dict
