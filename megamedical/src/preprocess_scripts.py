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
    for prefix in ["megamedical", "maxslice", "midslice"]:
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


def produce_slices(root_dir,
                   version,
                   dataset,
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

            if show_imgs and res==128:
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

                #place resized segs in regular array
                seg_res[..., lab_idx] = bin_seg_res

            if save:
                
                proc_dir = os.path.join(root_dir, f"megamedical_v{version}", dataset, mode, subject_name)
                if not os.path.exists(proc_dir):
                    os.makedirs(proc_dir)
                np.save(os.path.join(proc_dir, f"img_{res}.npy"), image_res)
                np.save(os.path.join(proc_dir, f"seg_{res}.npy"), seg_res)
                
                
def label_dist(dataset,
               proc_func,
               subdset,
               version,
               visualize,
               save):

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
        midslice_labels = {}
        for plane in planes:
            lab_shape = loaded_label.shape
            if len(lab_shape) == 2:
                midslice_labels[plane] = all_labels
            else:
                midslice_labels[plane] = np.unique(np.take(loaded_label, lab_shape[plane]//2, plane))
        return all_labels, midslice_labels
    
    proc_dir, label_info = proc_func(subdset,
                                   get_unique_labels,
                                   accumulate=True,
                                   version=version,
                                   save=save)
    
    total_label_info = [li[0] for li in label_info]
    flat_total_label_info = [label for subj in total_label_info for label in subj]
    frequency_label_dict = dict(Counter(flat_total_label_info))
    ax = sns.barplot(x=list(frequency_label_dict.keys()), y=list(frequency_label_dict.values()))
    if visualize:
        ax.bar_label(ax.containers[0])
        ax.set(title=f"Label Frequency for {dataset}/{subdset}")
        plt.show()
    if save:
        fig = ax.get_figure()
        save_dir = os.path.join(proc_dir, "figures")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig_dir = os.path.join(save_dir, "lab_info.png")
        fig.savefig(fig_dir)
