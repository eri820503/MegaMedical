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
                   loaded_image,
                   loaded_label,
                   dset_info,
                   save_slices=False, 
                   show_hists=False,
                   show_imgs=False,
                   subject_name=None):
       
    modality_names = dset_info["modality_names"]
    do_clip = dset_info["do_clip"]
    norm_scheme = dset_info["norm_scheme"]
    planes = dset_info["planes"]
    
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
            if norm_scheme=="CT":
                lower = clip_args[0]
                upper = clip_args[1]
            elif norm_scheme=="MR":
                lower = np.percentile(modality_loaded_image[modality_loaded_image>0], q=0.5)
                upper = np.percentile(modality_loaded_image[modality_loaded_image>0], q=99.5)
            else:
                raise ValueError("Normalization Scheme Not Implemented")
            modality_loaded_image = np.clip(modality_loaded_image, a_min=lower, a_max=upper)
        
        #normalize the volume
        normalized_modality_image = (modality_loaded_image - np.amin(modality_loaded_image))/(np.amax(modality_loaded_image) - np.amin(modality_loaded_image))
        
        if show_hists:
            display_histogram(normalized_modality_image.flatten())
       
        #Save the original image/seg pair
        if save_slices:
            save_name = image.split(".")[0]
            save_dir = os.path.join(new_mode_dir, save_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        
        #make the volume a 3D cube
        square_modality_image = squarify(normalized_modality_image)
        square_modality_label = squarify(loaded_label)
        
        #original square image size
        old_size = square_modality_image.shape[0]

        #Create standard deviation for 256 IMAGES/SEG and produce final 256 resized image
        sigma_256 = 1/4 * old_size / 256
        blurred_image_256 = ndimage.gaussian_filter(square_modality_image, sigma=sigma_256)
        zoom_tup_256 = (256/old_size, 256/old_size) if len(blurred_image_256.shape) == 2 else (256/old_size, 256/old_size, 256/old_size)
        final_resized_image_256 = ndimage.zoom(blurred_image_256, zoom=zoom_tup_256, order=1)
        
        #Create standard deviation for 128 IMAGES/SEG and produce final 128ized image
        sigma_128 = 1/4 * old_size / 128
        blurred_image_128 = ndimage.gaussian_filter(square_modality_image, sigma=sigma_128)
        zoom_tup_128 = (128/old_size, 128/old_size) if len(blurred_image_128.shape) == 2 else (128/old_size, 128/old_size, 128/old_size)
        final_resized_image_128 = ndimage.zoom(blurred_image_128, zoom=zoom_tup_128, order=1)
            
        #get all of the labels in the volume, without 0
        unique_labels = np.unique(loaded_label)[1:]
        
        #create the label matrix for accessing labels amounts
        if len(square_modality_image.shape) == 2:
            label_array_matrix = np.zeros([1, len(unique_labels)])
            
            final_resized_seg_256 = np.zeros((256, 256, len(unique_labels)))
            final_resized_seg_128 = np.zeros((128, 128, len(unique_labels)))
        else:
            #Note that this can be < or > than 256....
            #I don't know if this works -\_(:))/-
            label_array_matrix = np.zeros([3, np.amax([square_modality_label.shape[p] for p in planes]), len(unique_labels)])
            
            final_resized_seg_256 = np.zeros((256, 256, 256, len(unique_labels)))
            final_resized_seg_128 = np.zeros((128, 128, 128, len(unique_labels)))
        
        #Loop through all labels, construct final superlabel
        label_dict = {}
        for lab_idx, label in enumerate(unique_labels):
            label = int(label)
            label_dict[lab_idx] = label
            resized_binary_mask = np.float32((label==square_modality_label))
        
            #produce final 256 resized segmentation
            blurred_seg_256 = ndimage.gaussian_filter(resized_binary_mask, sigma=sigma_256)
            zoom_tup_256 = (256/old_size, 256/old_size) if len(blurred_seg_256.shape) == 2 else (256/old_size, 256/old_size, 256/old_size)
            resized_seg_256 = ndimage.zoom(blurred_seg_256, zoom=zoom_tup_256, order=0)
            
            #produce final 128 resized segmentation
            blurred_seg_128 = ndimage.gaussian_filter(resized_binary_mask, sigma=sigma_128)
            zoom_tup_128 = (128/old_size, 128/old_size) if len(blurred_seg_128.shape) == 2 else (128/old_size, 128/old_size, 128/old_size)
            resized_seg_128 = ndimage.zoom(blurred_seg_128, zoom=zoom_tup_128, order=0)
            
            #place resized segs in arrays
            final_resized_seg_256[..., lab_idx] = resized_seg_256
            final_resized_seg_128[..., lab_idx] = resized_seg_128
            
            for plane in planes:
                axes = [0,1,2]
                axes.remove(plane)
                ax = tuple(axes)
        
                if len(resized_binary_mask.shape) == 2:            
                    label_array_matrix[0, lab_idx] = np.count_nonzero(resized_binary_mask)
                else:                               
                    label_array_matrix[plane,:,lab_idx] = np.count_nonzero(resized_binary_mask, axis=ax)
                
                if show_imgs:
                    f, axarr = plt.subplots(nrows=1,ncols=2,figsize=[8,4])
                    
                    if len(final_resized_image_256.shape) == 2:
                        chosen_resized_slice = final_resized_image_256
                        chosen_resized_label = resized_seg_256
                    else:      
                        chosen_resized_slice = np.take(final_resized_image_256, 128, plane)
                        chosen_resized_label = np.take(resized_seg_256, 128, plane)
        
                    img_obj = axarr[0].imshow(chosen_resized_slice, interpolation="none", cmap="gray")
                    seg_obj = axarr[1].imshow(chosen_resized_label, interpolation="none")
                    plt.colorbar(img_obj, ax=axarr[0])
                    plt.colorbar(seg_obj, ax=axarr[1])
                    plt.show()
        
        if save_slices:
            save_name = image.split(".")[0]                                        
            new_image_dir = os.path.join(new_mode_dir, save_name)
            
            #Save the label dictionary
            label_dict_path = os.path.join(new_image_dir, "label_info_dict")
            dump_dictionary(label_dict, label_dict_path)
            
            #Save the processed images/segs
            resized_256_image_path = os.path.join(new_image_dir, "img_256.npy")
            resized_256_seg_path = os.path.join(new_image_dir, "seg_256.npy")
            resized_256_linfo_path = os.path.join(new_image_dir, "info_256.npy")
            
            resized_128_image_path = os.path.join(new_image_dir, "img_128.npy")
            resized_128_seg_path = os.path.join(new_image_dir, "seg_128.npy")
            resized_128_linfo_path = os.path.join(new_image_dir, "info_128.npy")
            
            old_size = label_array_matrix.shape[1]
            if label_array_matrix.shape[0] == 1:
                rla_256 = label_array_matrix
                rla_128 = label_array_matrix
            else:
                rla_256 = np.round(scipy.ndimage.interpolation.zoom(label_array_matrix, [1, 256/old_size, 1], order=1), 0)
                rla_128 = np.round(scipy.ndimage.interpolation.zoom(label_array_matrix, [1, 128/old_size, 1], order=1), 0)

            np.save(resized_256_image_path, final_resized_image_256)
            np.save(resized_256_seg_path, final_resized_seg_256)
            np.save(resized_256_linfo_path, rla_256)
            
            np.save(resized_128_image_path, final_resized_image_128)
            np.save(resized_128_seg_path, final_resized_seg_128)
            np.save(resized_128_linfo_path, rla_128)
