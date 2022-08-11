# universeg imports
import universeg as uvs

# random imports
import sys
import numpy as np
import os
import pathlib
import pickle
from tqdm import tqdm
import scipy
import math
import matplotlib.pyplot as plt
from scipy import ndimage
import shutil
import nibabel as nib
import nibabel.processing as nip
import glob


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def dump_dictionary(arr, file_name):
    with open(f'{file_name}.pickle', 'wb') as handle:
        pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def make_processed_dir(dset_name, dset_info, save_slices, dataset_ver="3.1", datasets_root_dir="/home/vib9/src/data/"):
    root_dir = os.path.join(datasets_root_dir, dset_info["main"], f"processed/megamedical_v3.1")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
            
    processed_dir = os.path.join(root_dir, dset_name)
    
    print("Starting:", processed_dir)

    if save_slices:
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
            
        for mode in dset_info["modality_names"]:
            new_mode_dir = os.path.join(processed_dir, mode)
            if not os.path.exists(new_mode_dir):
                os.makedirs(new_mode_dir)
    
    return processed_dir

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
            print("Before")
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
            print("After")
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
            resized_256_image_path = os.path.join(new_image_dir, f"{dataset_dir}|{mode}|{save_name}|resize_image_256.npy")
            resized_256_seg_path = os.path.join(new_image_dir, f"{dataset_dir}|{mode}|{save_name}|resize_seg_256.npy")
            resized_256_linfo_path = os.path.join(new_image_dir, f"{dataset_dir}|{mode}|{save_name}|resize_linfo_256.npy")
            
            resized_128_image_path = os.path.join(new_image_dir, f"{dataset_dir}|{mode}|{save_name}|resize_image_128.npy")
            resized_128_seg_path = os.path.join(new_image_dir, f"{dataset_dir}|{mode}|{save_name}|resize_seg_128.npy")
            resized_128_linfo_path = os.path.join(new_image_dir, f"{dataset_dir}|{mode}|{save_name}|resize_linfo_128.npy")
            
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


def get_all_datasets(dataset_ver, data_root="/home/vib9/src/data"):
    proper_dsets = []
    for dset in os.listdir(data_root):
        dset_dir = os.path.join(data_root, dset, f"processed/megamedical_v{dataset_ver}")
        if os.path.isdir(dset_dir):
            for sub_dset in os.listdir(dset_dir):
                proper_dsets.append(f"{dset}~{sub_dset}")
    return proper_dsets


def make_splits(datasets, dataset_ver, root="/share/sablab/nfs02/users/gid-dalcaav/data/v3.2-midslice"):
    for dataset in datasets:
        print(f"STARTING SPLITTING {dataset}")
        info = dataset.split("~")
        sub_dset_p = os.path.join(info[0], info[1])
        sub_dset_full_path = os.path.join(root, sub_dset_p)
        for modality in os.listdir(sub_dset_full_path):
            mod_dir = os.path.join(sub_dset_full_path, modality)
            save_names = np.array(list(filter(lambda val: val != "train.txt" and val !=
                                      "val.txt" and val != "test.txt", os.listdir(mod_dir))))
            #save_names = np.array(
            #    list(map(lambda val: os.path.join(sub_dset_p, modality, val), clean_names)))
            
            total_amount = len(save_names)
            indices = np.arange(total_amount)
            np.random.shuffle(indices)

            train_amount = int(total_amount*0.6)
            val_test_amount = total_amount - train_amount
            val_amount = int(val_test_amount*0.5)
            test_amount = val_test_amount - val_amount

            train_indices = indices[:train_amount]
            val_indices = indices[train_amount:train_amount+val_amount]
            test_indices = indices[-test_amount:]

            train_names = save_names[train_indices]
            val_names = save_names[val_indices]
            test_names = save_names[test_indices]
            
            #print(train_names)
            #raise ValueError
            
            train_file = open(os.path.join(mod_dir, "train.txt"), "w")
            val_file = open(os.path.join(mod_dir, "val.txt"), "w")
            test_file = open(os.path.join(mod_dir, "test.txt"), "w")

            for file_name in train_names:
                train_file.write(file_name + "\n")
            train_file.close()

            for file_name in val_names:
                val_file.write(file_name + "\n")
            val_file.close()

            for file_name in test_names:
                test_file.write(file_name + "\n")
            test_file.close()

        print(f"DONE SPLITTING {dataset}!")


def repackage_npz_to_npy(datasets, dataset_ver, root="/home/vib9/src/data"):
    for dataset in datasets:
        print(f"STARTING SMALLIFYING {dataset}")
        info = dataset.split("~")
        sub_dset_root = os.path.join(
            root, info[0], f"processed/megamedical_v{dataset_ver}", info[1])
        for modality in os.listdir(sub_dset_root):
            modality_subdset_root = os.path.join(sub_dset_root, modality)
            for path in os.listdir(modality_subdset_root):
                path_file = os.path.join(modality_subdset_root, path)
                path = os.path.join(path_file, f'{info[1]}|{modality}|{path}|resize_data_256.npz') 
                if os.path.isfile(path):
                    data_256 = np.load(path)
                    image_256 = data_256["image"]
                    seg_256 = data_256["seg"]
                    linfo_256 = data_256["linfo"]

                    data_128 = np.load(str(path).replace("256","128"))
                    image_128 = data_128["image"]
                    seg_128 = data_128["seg"]
                    linfo_128 = data_128["linfo"]

                    np.save(str(path).replace("256.npz","256_image.npy"), image_256)
                    np.save(str(path).replace("256.npz","256_seg.npy"), seg_256)
                    np.save(str(path).replace("256.npz","256_linfo.npy"), linfo_256)
                    np.save(str(path).replace("256.npz","128_image.npy"), image_128)
                    np.save(str(path).replace("256.npz","128_seg.npy"), seg_128)
                    np.save(str(path).replace("256.npz","128_linfo.npy"), linfo_128)

                    os.remove(path)
                    os.remove(str(path).replace("256","128"))

        print(f"DONE SWAPPING NPZ -> NPY {dataset}!")
        

def produce_new_res(datasets, dataset_ver, root ="/share/sablab/nfs02/users/gid-dalcaav/data/v3.2-midslice"):
    for dataset in datasets:
        print(f"STARTING MIDIFYING {dataset}")
        info = dataset.split("~")
        sub_dset_root = os.path.join(root, info[0], info[1])
        for modality in os.listdir(sub_dset_root):
            modality_subdset_root = os.path.join(sub_dset_root, modality)
            for path in os.listdir(modality_subdset_root):
                if path not in ["train.txt", "val.txt", "test.txt"]:
                    path_file = os.path.join(modality_subdset_root, path)
                    
                    img_128_name = glob.glob(os.path.join(path_file, "resize_image_128_*"))[0]
                    seg_128_name = glob.glob(os.path.join(path_file, "resize_seg_128_*"))[0]

                    image_128_volume = np.load(img_128_name, mmap_mode='r')
                    seg_128_volume = np.load(seg_128_name, mmap_mode='r')
                    
                    """
                    f, axarr = plt.subplots(nrows=1, ncols=2)
                    if seg_128_volume.shape[2] != 1:
                        axarr[0].imshow(image_128_volume)
                        axarr[1].imshow(np.argmax(seg_128_volume, axis=2))
                    else:
                        axarr[0].imshow(image_128_volume)
                        axarr[1].imshow(seg_128_volume)
                        
                    plt.show()
                    """

                    image_sigma_64 = (0.5, 0.5)
                    seg_sigma_64 = (0.5, 0.5, 0)

                    blurred_image_128 = ndimage.gaussian_filter(image_128_volume, sigma=image_sigma_64)
                    blurred_seg_128 = ndimage.gaussian_filter(seg_128_volume, sigma=seg_sigma_64)

                    image_zoom_tup_64 = (0.5, 0.5)
                    seg_zoom_tup_64 = (0.5, 0.5, 1)

                    img_64_volume = ndimage.zoom(blurred_image_128, zoom=image_zoom_tup_64, order=1)
                    seg_64_volume = ndimage.zoom(blurred_seg_128, zoom=seg_zoom_tup_64, order=0)
                    
                    """
                    f, axarr = plt.subplots(nrows=1, ncols=2)
                    if seg_128_volume.shape[2] != 1:
                        axarr[0].imshow(img_64_volume)
                        axarr[1].imshow(np.argmax(seg_64_volume, axis=2))
                    else:
                        axarr[0].imshow(img_64_volume)
                        axarr[1].imshow(seg_64_volume)
                    
                    plt.show()
                    """
                    def repl_last(s, sub, repl):
                        index = s.rfind(sub)
                        if index == -1:
                            return s
                        return s[:index] + repl + s[index+len(sub):]

                    img_64_name = repl_last(img_128_name, "128", "64")
                    seg_64_name = repl_last(seg_128_name, "128", "64")

                    #print(img_64_name)
                    #print(seg_64_name)
                
                    np.save(img_64_name, img_64_volume)
                    np.save(seg_64_name, seg_64_volume)
                
        print(f"DONE MIDIFYING {dataset}!")
        

def produce_pickl_files(datasets, dataset_ver, root ="/home/vib9/src/data"):
    for dataset in datasets:
        try:
            print(f"STARTING TRANSFERING PIKL FOR {dataset}")
            info = dataset.split("~")
            sub_dset_root = os.path.join(root, info[0], f"processed/megamedical_v{dataset_ver}", info[1])
            midslice_dset_root = os.path.join(root, info[0], f"processed/midslice_v{dataset_ver}", info[1])

            for modality in os.listdir(midslice_dset_root):

                modality_subdset_root = os.path.join(sub_dset_root, modality)
                midslice_mod_root = os.path.join(midslice_dset_root, modality)

                for path in os.listdir(midslice_mod_root):
                    if not(path in ["train.txt", "val.txt", "test.txt"]):
                        path_file = os.path.join(modality_subdset_root, path)
                        new_midslice_file = os.path.join(midslice_mod_root, path)

                        pickle_path = os.path.join(new_midslice_file, "label_info_dict.pickle")
                        new_pickle_path = os.path.join(path_file, "label_info_dict.pickle")
                        if not os.path.exists(new_pickle_path):
                            shutil.copyfile(pickle_path, new_pickle_path)
        except:
            print(f"Error in {dataset}")
               
        print(f"DONE TRANSFERING PIKL FOR {dataset}!")
        

def check_processed(datasets, dataset_ver, root="/home/vib9/src/data"):
    invalid_subdsets = []
    missing_files = 0
    for dataset in datasets:
        info = dataset.split("~")
        sub_dset_root = os.path.join(root, info[0], f"processed/megamedical_v{dataset_ver}", info[1])
        for modality in os.listdir(sub_dset_root):
            modality_subdset_root = os.path.join(sub_dset_root, modality)
            clean_names = list(filter(lambda val: val != "train.txt" and val !=
                                      "val.txt" and val != "test.txt", os.listdir(modality_subdset_root)))
            for path in clean_names:
                path_file = os.path.join(modality_subdset_root, path)
                path = os.path.join(path_file, f'{info[1]}|{modality}|{path}|resize_data_256.npz')
                
                img_256_path = str(path).replace("256.npz","256_image.npy")
                seg_256_path = str(path).replace("256.npz","256_seg.npy")
                linfo_256_path = str(path).replace("256.npz","256_linfo.npy")
                
                img_128_path = str(path).replace("256.npz","128_image.npy")
                seg_128_path = str(path).replace("256.npz","128_seg.npy")
                linfo_128_path = str(path).replace("256.npz","128_linfo.npy") 
                
                files = [img_256_path, seg_256_path, linfo_256_path, img_128_path, seg_128_path, linfo_128_path]
                for file in files:
                    if not os.path.isfile(file):
                        missing_files += 1
                        print(f"Missing!: {file}")
                        if not dataset in invalid_subdsets:
                            invalid_subdsets.append(dataset)
                            break
                            
        print(f"PROBLEMATIC DATASETS: {invalid_subdsets}!, MISSING FILES: {missing_files}")


def process_datasets(dataset_ver, resize, generate_splits, repackage_files, check_dsets, mid_datasets, ppf, list_to_proc):
    assert dataset_ver is not None, "Must choose a version manually."
    if generate_splits:
        make_splits(list_to_proc, dataset_ver)
    if repackage_files:
        repackage_npz_to_npy(list_to_proc, dataset_ver)
    if check_dsets:
        check_processed(list_to_proc, dataset_ver)
    if mid_datasets:
        produce_midslice_dataset(list_to_proc, dataset_ver)
    if ppf:
        produce_pickl_files(list_to_proc, dataset_ver)
    if resize:
        produce_new_res(list_to_proc, dataset_ver)


if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        dataset = sys.argv[1]
        ltp = []
        for sub_dataset in os.listdir(os.path.join("/share/sablab/nfs02/users/gid-dalcaav/data/v3.2-midslice",dataset)):
            ltp.append(f"{dataset}~{sub_dataset}")
    else:
        #ltp = get_all_datasets(dataset_ver="3.1")
        # Test dataset
        ltp = [""]
    
    dataset_ver = None

    process_datasets(dataset_ver,
                     resize=False,
                     generate_splits=True,
                     repackage_files=False,
                     check_dsets=False,
                     mid_datasets=False,
                     ppf=False,
                     list_to_proc=ltp)
