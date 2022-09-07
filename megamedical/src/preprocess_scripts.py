# random imports
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from megamedical.utils.registry import paths
from megamedical.utils.proc_utils import *

def make_processed_dir(dset, subdset, save, dataset_ver, dset_info):
    root_dir = os.path.join(paths["DATA"], dset, f"processed")
    
    dirs_to_make = []
    for prefix in ["maxslice", "midslice"]:
        dirs_to_make.append(os.path.join(root_dir, f"{prefix}_v{dataset_ver}", subdset))

    if save:
        for dtm in dirs_to_make:
            if not os.path.exists(dtm):
                os.makedirs(dtm)
        for mode in dset_info["modality_names"]:
            for subset in dirs_to_make:
                mode_dir = os.path.join(subset, mode)
                if not os.path.exists(mode_dir):
                    os.makedirs(mode_dir)
    return root_dir


def save_maxslice(proc_dir, image_res, seg_res, res, planes, maxslices):
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)
        
    for plane in planes:     
        maxslice_img_dir = os.path.join(proc_dir, f"img_{res}_{plane}.npy")
        maxslice_seg_dir = os.path.join(proc_dir, f"seg_{res}_{plane}.npy")
        # Get midslice slices
        subj_shape = image_res.shape
        if len(subj_shape) == 2:
            maxslice_img = image_res
            maxslice_seg = seg_res
        else:
            slices = maxslices[plane]
            maxslice_img = np.stack([np.take(image_res, int(si), plane) for si in slices], -1)
            maxslice_seg = np.stack([np.take(seg_res, int(si), plane) for si in slices], -1)
        np.save(maxslice_img_dir, maxslice_img)
        np.save(maxslice_seg_dir, maxslice_seg)


def save_midslice(proc_dir, image_res, seg_res, res, planes):
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)
    
    for plane in planes:   
        midslice_img_dir = os.path.join(proc_dir, f"img_{res}_{plane}.npy")
        midslice_seg_dir = os.path.join(proc_dir, f"seg_{res}_{plane}.npy")
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
                   resolutions=[256,128],
                   save=False,
                   show_hists=False,
                   show_imgs=False):
    # Set the name to be saved
    save_name = subject_name.split(".")[0]
    
    # Get all of the labels in the volume population
    unique_labels = np.load(os.path.join(root_dir, "label_info", subdset, "all_labels.npy"))
    
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
            #Resize to several resolutions
            image_res = blur_and_resize(square_image, old_size, new_size=res, order=1)

            #final segmentations are with labels in the last dimension
            if len(square_image.shape) == 2:
                seg_res = np.zeros((res, res, len(unique_labels)))
            else:     
                seg_res = np.zeros((res, res, res, len(unique_labels)))

            #go through unique labels and add to slices
            max_slices = {pl:np.zeros((len(unique_labels))) for pl in dset_info["planes"]}
            for lab_idx, label in enumerate(unique_labels):
                #isolate mask of label
                label = int(label)
                bin_mask = np.float32((square_label==label))
                
                #produce resized segmentations
                bin_seg_res = blur_and_resize(bin_mask, old_size, new_size=res, order=0)
                
                # Gather maxslice info
                if len(bin_seg_res.shape) == 3:
                    for pl in dset_info["planes"]:
                        all_axes = [0,1,2]
                        all_axes.remove(pl)
                        greatest_index = np.argmax(np.count_nonzero(bin_seg_res, axis=tuple(all_axes)))
                        max_slices[pl][lab_idx] = greatest_index

                # Place resized segs in regular array
                seg_res[..., lab_idx] = bin_seg_res
            
            if save:
                #Save maxslice files
                maxslice_proc_dir = os.path.join(root_dir, f"maxslice_v{version}", subdset, mode, subject_name)
                save_maxslice(maxslice_proc_dir, image_res, seg_res, res, dset_info["planes"], max_slices)
                #Save midslice files
                midslice_proc_dir = os.path.join(root_dir, f"midslice_v{version}", subdset, mode, subject_name)
                save_midslice(midslice_proc_dir, image_res, seg_res, res, dset_info["planes"])
                
                
def label_dist(dataset,
               proc_func,
               subdset,
               version,
               visualize,
               save,
               volume_wide):

    def get_unique_labels(proc_dir,
                          version,
                          dset_name,
                          image, 
                          loaded_image,
                          loaded_label,
                          dset_info,
                          show_hists,
                          show_imgs,
                          save):
        planes = dset_info["planes"]
        all_labels = np.unique(loaded_label)
        all_labels = np.delete(all_labels, [0])
        
        maxslice_labels = {}
        midslice_labels = {}
        for plane in planes:
            lab_shape = loaded_label.shape
            if len(lab_shape) == 2:
                maxslice_labels[plane] = all_labels
                midslice_labels[plane] = all_labels
            else:
                all_axis = [0, 1, 2]
                all_axis.remove(plane)
                slice_idx = np.argmax(np.sum(loaded_label, axis=tuple(all_axis)))
                max_unique_labels = np.unique(np.take(loaded_label, slice_idx, plane))
                mid_unique_labels = np.unique(np.take(loaded_label, lab_shape[plane]//2, plane))
                maxslice_labels[plane] = np.delete(max_unique_labels, [0])
                midslice_labels[plane] = np.delete(mid_unique_labels, [0])
                
        return all_labels, maxslice_labels, midslice_labels, planes
    
    proc_dir, label_info = proc_func(subdset,
                                     get_unique_labels,
                                     load_images=False,
                                     accumulate=True,
                                     version=version,
                                     save=save)
    planes = label_info[0][3]
    num_subjects = len(label_info)
    total_label_info = [li[0] for li in label_info]
    maxslice_label_info = [li[1] for li in label_info]
    midslice_label_info = [li[2] for li in label_info]
    
    if volume_wide:
        flat_total_label_info = [label for subj in total_label_info for label in subj]
        frequency_label_dict = dict(Counter(flat_total_label_info))
        
        if save or visualize:
            sns.set(rc = {'figure.figsize':(30,5)})
            ax = sns.barplot(x=list(frequency_label_dict.keys()), y=list(frequency_label_dict.values()))
            ax.bar_label(ax.containers[0])
            ax.set(title=f"Label Frequency for {dataset}/{subdset} of {num_subjects} many subjects.")
            plt.show()
            if save:
                fig = ax.get_figure()
                save_dir = os.path.join(proc_dir, "figures", dataset)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                fig_dir = os.path.join(save_dir, "lab_info.png")
                dict_dir = os.path.join(save_dir, "lab_dict")
                dump_dictionary(frequency_label_dict, dict_dir)
                fig.savefig(fig_dir)
    else:
        total_midslice_label_dict = {}
        total_maxslice_label_dict = {}
        for plane in planes:
            mid_plane_unique_labels = [mid_info[plane] for mid_info in midslice_label_info]
            flat_mid_plane_labels = [label for subj in mid_plane_unique_labels for label in subj]
            total_midslice_label_dict[plane] = dict(Counter(flat_mid_plane_labels))
            
            max_plane_unique_labels = [max_info[plane] for max_info in maxslice_label_info]
            flat_max_plane_labels = [label for subj in max_plane_unique_labels for label in subj]
            total_maxslice_label_dict[plane] = dict(Counter(flat_max_plane_labels))
        
        sns.set(rc = {'figure.figsize':(30,5)})
        for plane in planes:
            if visualize:
                ax1 = sns.barplot(x=list(total_midslice_label_dict[plane].keys()), y=list(total_midslice_label_dict[plane].values()))
                ax1.set(title=f"Midslice | Plane {plane} | Label Frequency for {dataset}/{subdset} of {num_subjects} many subjects.")
                ax1.bar_label(ax1.containers[0])
                plt.show()
                ax2 = sns.barplot(x=list(total_maxslice_label_dict[plane].keys()), y=list(total_maxslice_label_dict[plane].values()))
                ax2.set(title=f"Maxslice | Plane {plane} | Label Frequency for {dataset}/{subdset} of {num_subjects} many subjects.")
                ax2.bar_label(ax2.containers[0])
                plt.show()
            if save:
                save_dir = os.path.join(proc_dir, "figures", subdset)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                mid_dict_dir = os.path.join(save_dir, f"mid_lab_dict_{plane}")
                max_dict_dir = os.path.join(save_dir, f"max_lab_dict_{plane}")
                dump_dictionary(total_midslice_label_dict, mid_dict_dir)
                dump_dictionary(total_maxslice_label_dict, max_dict_dir)
                
                
def label_info(data_obj,
               subdset,
               version,
               save):

    def get_label_amounts(proc_dir,
                          version,
                          dset_name,
                          image, 
                          loaded_image,
                          loaded_label,
                          dset_info,
                          show_hists,
                          show_imgs,
                          save):
        planes = dset_info["planes"]
        all_labels = np.unique(loaded_label)
        all_labels = np.delete(all_labels, [0])
        total_lab_amount_dict = {lab : np.count_nonzero((loaded_label==lab).astype(int)) for lab in all_labels}
        
        maxslice_amount_dict = {}
        midslice_amount_dict = {}
        lab_shape = loaded_label.shape
        if len(lab_shape) == 2:
            midslice_amount_dict[0] = total_lab_amount_dict
            maxslice_amount_dict[0] = total_lab_amount_dict
        else:
            for plane in planes:
                all_axes = [0,1,2]
                all_axes.remove(plane)
                
                midslice = np.take(loaded_label, lab_shape[plane]//2, plane)
                mid_unique_labels = np.unique(midslice)
                midslice_plane_labels = np.delete(mid_unique_labels, [0])
                
                midslice_amount_dict[plane] = {lab : np.count_nonzero((midslice==lab).astype(int)) for lab in midslice_plane_labels}
                maxslice_amount_dict[plane] = {lab : np.amax(np.count_nonzero((loaded_label==lab).astype(int), axis=tuple(all_axes))) for lab in all_labels}
        
        return maxslice_amount_dict, midslice_amount_dict, all_labels
    
    proc_dir, label_info = data_obj.proc_func(subdset,
                                     get_label_amounts,
                                     load_images=False,
                                     accumulate=True,
                                     version=version,
                                     save=save)

    maxslice_label_info = [li[0] for li in label_info]
    midslice_label_info = [li[1] for li in label_info]
    total_label_info = [li[2] for li in label_info]
    unique_labels = list(set([label for subj in total_label_info for label in subj]))
    
    save_dir = os.path.join(proc_dir, "label_info", subdset)
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        all_label_dir = os.path.join(save_dir, f"all_labels")
        np.save(all_label_dir, np.array(unique_labels))
    
    for plane in data_obj.dset_info[subdset]["planes"]:
        # +1 accounts for starting from zero error
        max_label_info_array = np.zeros((len(total_label_info), len(unique_labels) + 1))
        mid_label_info_array = np.zeros((len(total_label_info), len(unique_labels) + 1))
        for subj_idx, (max_info, mid_info) in enumerate(zip(maxslice_label_info, midslice_label_info)):
            for max_label in max_info[plane].keys():
                max_label_info_array[subj_idx, int(max_label)] = max_info[plane][max_label]
            for mid_label in mid_info[plane].keys():
                mid_label_info_array[subj_idx, int(mid_label)] = mid_info[plane][mid_label]
        if save:
            mid_dict_dir = os.path.join(save_dir, f"midslice_pop_lab_amount_{plane}")
            max_dict_dir = os.path.join(save_dir, f"maxslice_pop_lab_amount_{plane}")
            np.save(mid_dict_dir, mid_label_info_array)
            np.save(max_dict_dir, max_label_info_array)

