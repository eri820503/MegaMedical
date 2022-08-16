# random imports
import numpy as np
import os
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

from megamedical.utils.registry import paths
from megamedical.utils.proc_utils import *

def make_processed_dir(dset, subdset, save_slices, dataset_ver):
    root_dir = os.path.join(paths["DATA"], dset, f"processed/{dataset_ver}")   
    processed_dir = os.path.join(root_dir, subdset)
    print("Starting:", processed_dir)

    if save_slices:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
       
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
            
        for mode in dset_info["modality_names"]:
            new_mode_dir = os.path.join(processed_dir, mode)
            if not os.path.exists(new_mode_dir):
                os.makedirs(new_mode_dir)
    
    return processed_dir


def produce_slices(root_dir,
                   dataset,
                   subject_name,
                   loaded_image,
                   loaded_label,
                   dset_info,
                   save_slices=False, 
                   show_hists=False,
                   show_imgs=False,
                   subject_name=None):
    # Set the name to be saved
    save_name = subject_name.split(".")[0]
    
    # Unpack some things from the info dictionary
    modality_names = dset_info["modality_names"]
    do_clip = dset_info["do_clip"]
    norm_scheme = dset_info["norm_scheme"]
    planes = dset_info["planes"]
    
    #get all of the labels in the volume, once 
    unique_labels = np.unique(loaded_label)[1:]
    
    for idx, mode in enumerate(modality_names):
        new_mode_dir = os.path.join(root_dir, mode)
        
        #Extract the modality if it exists (mostly used for MSD)
        if len(modality_names) != 1:
            modality_loaded_image = loaded_image[:,:,:,idx]
        else:
            modality_loaded_image = loaded_image
         
        if show_hists:
            display_histogram(modality_loaded_image.flatten())
            
        if do_clip:
            #clip volume between prespecified values
            modality_loaded_image = clip_volume(modality_loaded_image, norm_scheme, clip_args)
        
        #normalize the volume between [0,1]
        normalized_modality_image = relative_norm(modality_loaded_image)
        
        if show_hists:
            display_histogram(normalized_modality_image.flatten())
       
        #Save the original image/seg pair
        if save_slices:
            save_dir = os.path.join(new_mode_dir, save_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        
        #make the volume/label a 3D cube
        square_image = squarify(normalized_modality_image)
        square_label = squarify(loaded_label)
        
        #original square image size
        old_size = square_image.shape[0]

        #Resize to several resolutions
        image_256 = blur_and_resize(square_image, old_size, new_size=256, order=1)
        image_128 = blur_and_resize(square_image, old_size, new_size=128, order=1)
        
        #final segmentations are with labels in the last dimension
        if len(square_modality_image.shape) == 2:
            seg_256 = np.zeros((256, 256, len(unique_labels)))
            seg_128 = np.zeros((128, 128, len(unique_labels)))
        else:     
            seg_256 = np.zeros((256, 256, 256, len(unique_labels)))
            seg_128 = np.zeros((128, 128, 128, len(unique_labels)))

        
        for lab_idx, label in enumerate(unique_labels):
            label = int(label)
            resized_binary_mask = np.float32((label==square_modality_label))
        
            #produce resized segmentations
            seg_256 = blur_and_resize(resized_binary_mask, old_size, new_size=256, order=0)
            seg_128 = blur_and_resize(resized_binary_mask, old_size, new_size=128, order=0)
            
            #place resized segs in arrays
            final_resized_seg_256[..., lab_idx] = resized_seg_256
            final_resized_seg_128[..., lab_idx] = resized_seg_128
            
            for plane in planes:
                axes = [0,1,2]
                axes.remove(plane)
                ax = tuple(axes)
                
                if show_imgs:
                    display_processing_slices(final_resized_image_256, resized_seg_256)
        
        if save_slices:
            save_name = image.split(".")[0]                                        
            new_image_dir = os.path.join(new_mode_dir, save_name)
            
            # Save 256 resolution
            np.save(os.path.join(new_image_dir, "img_256.npy"), final_resized_image_256)
            np.save(os.path.join(new_image_dir, "seg_256.npy"), final_resized_seg_256)
            
            # Save 128 resolution
            np.save(os.path.join(new_image_dir, "img_128.npy"), final_resized_image_128)
            np.save(os.path.join(new_image_dir, "seg_128.npy"), final_resized_seg_128)
            
            # Save 64 resolution
            np.save(os.path.join(new_image_dir, "img_64.npy"), final_resized_image_64)
            np.save(os.path.join(new_image_dir, "seg_64.npy"), final_resized_seg_64)
