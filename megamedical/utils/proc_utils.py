import pickle
import glob
import nibabel as nib
import nibabel.processing as nip
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import scipy

# returns True if not both midslice and maxslice are processed.
def check_proc_res(item):
    resolutions_to_process = []
    if item["task"] == "images":
        for dt in ["midslice", "maxslice"]:
            for res in item['resolutions']:
                template_root = os.path.join(item['proc_dir'], f"res{res}", item['dataset'])
                if len(glob.glob(os.path.join(template_root, f"{dt}_v{item['version']}", item['subdset'], "*/*", item['image']))) == 0:
                    resolutions_to_process.append(res)
        
    elif item["task"] == "stats":
        for res in item['resolutions']:
                template_root = os.path.join(item['proc_dir'], f"res{res}", item['dataset'], "label_info", item['subdset'], "pop_info_files")
                if not os.path.exists(os.path.join(template_root, f"{item['image']}.pickle")):
                    resolutions_to_process.append(res)
        
    return list(set(resolutions_to_process))
            

def get_list_of_subjects(root,
                         res,
                         data_type,
                         dset,
                         subdset):
    # Get the number of subjects by the first modality and first plane
    subdset_dir = os.path.join(root, f"res{res}", dset, data_type, subdset)
    modality_dir = os.path.join(subdset_dir, os.listdir(subdset_dir)[0])
    plane_dir = os.path.join(modality_dir, os.listdir(modality_dir)[0])
    return os.listdir(plane_dir)


def get_label_amounts(proc_dir,
                      version,
                      subdset,
                      save_name, 
                      loaded_image,
                      loaded_label,
                      dset_info,
                      show_hists,
                      show_imgs,
                      resolutions,
                      save):
    
    square_label = squarify(loaded_label)
    
    for res in resolutions:

        # Handle resizing
        old_seg_size = square_label.shape[0]
        ratio = res/old_seg_size

        if len(loaded_label.shape) == 2:
            zoom_tup = (ratio, ratio)
        else:
            zoom_tup = (ratio, ratio, ratio)

        # Get all labels at this resolution
        lab_info_root = os.path.join(proc_dir, f"res{res}", dset_info["main"], "label_info", subdset)
        all_labels = np.load(os.path.join(lab_info_root, "all_labels.npy"))   
            
        # Create statistics
        midslice_amounts = {plane : np.zeros(len(all_labels)) for plane in dset_info["planes"]}
        maxslice_amounts = {plane : np.zeros(len(all_labels)) for plane in dset_info["planes"]}
        
        for l_idx, lab in enumerate(all_labels):
            bin_mask = np.float32(square_label==lab)

            #produce resized segmentations
            bin_seg_res = blur_and_resize(bin_mask, old_seg_size, new_size=res, order=0, blur=False)

            for plane in dset_info["planes"]:
                if len(square_label.shape) == 2:
                    midslice_amounts[plane][l_idx] = np.mean(bin_seg_res)
                    maxslice_amounts[plane][l_idx] = np.mean(bin_seg_res)
                else:
                    all_axes = [0,1,2]
                    all_axes.remove(plane)
                    midslice_amounts[plane][l_idx] = np.round(np.mean(np.take(bin_seg_res, bin_seg_res.shape[plane]//2, plane)), 5)                 
                    maxslice_amounts[plane][l_idx] = np.amax(np.round(np.mean(bin_seg_res, axis=tuple(all_axes)), 5))
        if save:
            # Save dir for all the pickle files
            save_dir = os.path.join(lab_info_root, "pop_info_files")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dict = {
                "midslice": midslice_amounts,
                "maxslice": maxslice_amounts
            }
            subj_file = os.path.join(save_dir, save_name) 
            dump_dictionary(save_dict, subj_file)


def get_all_unique_labels(proc_dir,
                          version,
                          dset_name,
                          save_name, 
                          loaded_image,
                          loaded_label,
                          dset_info,
                          show_hists,
                          show_imgs,
                          resolutions,
                          save):
    
    # Get all labels and get rid of 0
    all_labels = np.delete(np.unique(loaded_label), [0])
    res_dict = {res: all_labels for res in resolutions}
    return res_dict


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


def relative_norm(image):
    return (image - np.amin(image))/(np.amax(image) - np.amin(image))


def display_processing_slices(image, seg, plane):
    f, axarr = plt.subplots(nrows=1,ncols=2,figsize=[8,4])
                    
    if len(image.shape) == 2:
        image_slice = image
        label_slice = seg
    else:      
        all_axis = [0, 1, 2]
        all_axis.remove(plane)
        slice_idx = np.argmax(np.sum(seg, axis=tuple(all_axis)))
        image_slice = np.take(image, slice_idx, plane)
        label_slice = np.take(seg, slice_idx, plane)

    img_obj = axarr[0].imshow(image_slice, interpolation="none", cmap="gray")
    seg_obj = axarr[1].imshow(label_slice, interpolation="none")
    plt.colorbar(img_obj, ax=axarr[0])
    plt.colorbar(seg_obj, ax=axarr[1])
    plt.show()

def blur_and_resize(image, old_size, new_size, order, blur=True):
    sigma = 1/4 * old_size / new_size
    ratio = new_size/old_size
    
    if blur:
        blurred_resized_image = ndimage.gaussian_filter(image, sigma=sigma)
    else:
        blurred_resized_image = image
    
    if len(blurred_resized_image.shape) == 2:
        zoom_tup = (ratio, ratio)
    else:
        zoom_tup = (ratio, ratio, ratio)
        
    resized_image = ndimage.zoom(blurred_resized_image, zoom=zoom_tup, order=order)
    return resized_image


def clip_volume(image, norm_scheme, clip_args):
    if norm_scheme=="CT":
        lower = clip_args[0]
        upper = clip_args[1]
        clipped_image = np.clip(image, a_min=lower, a_max=upper)
    elif norm_scheme=="MR":
        # 0.5 - 99.5
        lower = np.percentile(image[image>0], q=clip_args[0])
        upper = np.percentile(image[image>0], q=clip_args[1])
        clipped_image = np.clip(image, a_min=lower, a_max=upper)
    else:
        clipped_image = image
    return clipped_image


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

    
def dump_dictionary(arr, file_name):
    with open(f'{file_name}.pickle', 'wb') as handle:
        pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def squarify(M):
    if len(M.shape) == 2:
        (a,b)=M.shape
        if np.amax(M.shape)==a:
            b_dif = a-b
            
            b_left_pad = math.floor(b_dif/2)
            b_right_pad = math.ceil(b_dif/2)
            
            padding=((0,0),(b_left_pad,b_right_pad))
        else:
            a_dif = b-a
            
            a_left_pad = math.floor(a_dif/2)
            a_right_pad = math.ceil(a_dif/2)
            
            padding=((a_left_pad,a_right_pad),(0,0))
            
        return np.pad(M,padding,mode='constant',constant_values=0)
    elif len(M.shape) == 3:
        (a,b,c)=M.shape
        if np.amax(M.shape)==a:
            b_dif = a-b
            c_dif = a-c
            
            b_left_pad = math.floor(b_dif/2)
            b_right_pad = math.ceil(b_dif/2)
            
            c_left_pad = math.floor(c_dif/2)
            c_right_pad = math.ceil(c_dif/2)
            
            padding=((0,0),(b_left_pad,b_right_pad),(c_left_pad,c_right_pad))
            
        elif np.amax(M.shape)==b:
            a_dif = b-a
            c_dif = b-c
            
            a_left_pad = math.floor(a_dif/2)
            a_right_pad = math.ceil(a_dif/2)
            
            c_left_pad = math.floor(c_dif/2)
            c_right_pad = math.ceil(c_dif/2)
            
            padding=((a_left_pad,a_right_pad),(0,0),(c_left_pad,c_right_pad))
            
        else:
            a_dif = c-a
            b_dif = c-b
            
            a_left_pad = math.floor(a_dif/2)
            a_right_pad = math.ceil(a_dif/2)
            
            b_left_pad = math.floor(b_dif/2)
            b_right_pad = math.ceil(b_dif/2)
            
            padding=((a_left_pad,a_right_pad),(b_left_pad,b_right_pad),(0,0))
            
        return np.pad(M,padding,mode='constant',constant_values=0)
    else:
        raise ValueError("Improper shape for padding.")
        
        
def display_histogram(vol):
    plt.hist(vol, bins=20)
    plt.show()
    
    
def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing
    
    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation
    
    Returns:
    ----------
    new_img: The resampled nibabel image 
    
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    return new_img


def resample_mask_to(msk, to_img):
    """Resamples the nifti mask from its original spacing to a new spacing specified by its corresponding image
    
    Parameters:
    ----------
    msk: The nibabel nifti mask to be resampled
    to_img: The nibabel image that acts as a template for resampling
    
    Returns:
    ----------
    new_msk: The resampled nibabel mask 
    
    """
    to_img.header['bitpix'] = 8
    to_img.header['datatype'] = 2  # uint8
    new_msk = nib.processing.resample_from_to(msk, to_img, order=0)
    return new_msk