import pickle
import nibabel as nib
import nibabel.processing as nip
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy

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

def blur_and_resize(image, old_size, new_size, order):
    sigma = 1/4 * old_size / new_size
    ratio = new_size/old_size
    
    blurred_resized_image = ndimage.gaussian_filter(image, sigma=sigma)
    
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
    elif norm_scheme=="MR":
        # 0.5 - 99.5
        lower = np.percentile(image[image>0], q=clip_args[0])
        upper = np.percentile(image[image>0], q=clip_args[1])
    else:
        raise ValueError("Normalization Scheme Not Implemented")
    clipped_image = np.clip(image, a_min=lower, a_max=upper)
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