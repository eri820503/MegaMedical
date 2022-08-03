import numpy as np
import nibabel as nib
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
import pickle
from PIL import Image
from glob import glob
import SimpleITK as sitk
import imageio as io
import nrrd
import cv2
import gzip
import scipy
import pathlib
import glob
import medpy.io


class CannabisT1:

    def __init__(self):
        
        self.dataset_info_dictionary = {
            "retreived_2018_01_30":{
                "main":"CannabisT1",
                "image_root_dir":"/home/vib9/src/data/CannabisT1/processed/original_unzipped/retreived_2018_01_30/CannabisStudy/subjects",
                "label_root_dir":"/home/vib9/src/data/CannabisT1/processed/original_unzipped/retreived_2018_01_30/CannabisStudy/subjects",
                "modality_names":["T1"],
                "planes":[0, 1, 2],
                "clip_args": None,
                "norm_scheme": "MR",
                "do_clip": False,
                "proc_size":256
            }
        }


    def produce_slices(self,
                       root_dir,
                       dataset_dir,
                       loaded_image,
                       loaded_label,
                       modality_names,
                       image,
                       planes,
                       select_labels=None, 
                       proc_size=256,
                       save_slices=False, 
                       show_hists=False,
                       show_imgs=False,
                       do_resize=True,
                       do_clip=False,
                       clip_args=None,
                       norm_scheme=None):

        for idx, mode in enumerate(modality_names):
            new_mode_dir = os.path.join(root_dir, mode)

            #Extract the modality if it exists (mostly used for MSD)
            if len(modality_names) != 1:
                modality_loaded_image = loaded_image[idx,:,:,:]
            else:
                modality_loaded_image = loaded_image

            if show_hists:
                print("Before")
                display_historgram(modality_loaded_image.flatten())

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
                display_historgram(normalized_modality_image.flatten())

            #get all of the labels in the volume, without 0
            unique_labels = np.unique(loaded_label) if select_labels is None else select_labels

            #save original volume too
            original_normalized_image = normalized_modality_image
            original_normalized_label = loaded_label
            #resize the volume to (256,256)

            if do_resize:
                #make the volume a 3D cube
                square_modality_image = squarify(normalized_modality_image)
                square_modality_label = squarify(loaded_label)

                ssize = square_modality_image.shape[0]

                if len(square_modality_image.shape) == 2:
                    z = (proc_size/ssize, proc_size/ssize)
                elif len(square_modality_image.shape) == 3:
                    z = (proc_size/ssize, proc_size/ssize, proc_size/ssize)
                else:
                    raise ValueError("Improper square image shape!")

                resized_modality_image = ndimage.zoom(square_modality_image,zoom=z,order=1)
                resized_modality_label = ndimage.zoom(square_modality_label,zoom=z,order=0)
            else:
                resized_modality_image = original_normalized_image
                resized_modality_label = original_normalized_label

            if save_slices:
                save_name = image.split(".")[0]                                      
                if not os.path.isdir(os.path.join(new_mode_dir,save_name)):
                    os.makedirs(os.path.join(new_mode_dir,save_name))
                new_image_dir = os.path.join(new_mode_dir,save_name)
                np.save(os.path.join(new_image_dir,f"{dataset_dir}|{mode}|{save_name}|proc_volume"), original_normalized_image)
                np.save(os.path.join(new_image_dir,f"{dataset_dir}|{mode}|{save_name}|proc_label"), original_normalized_label)
                np.save(os.path.join(new_image_dir,f"{dataset_dir}|{mode}|{save_name}|resize_volume"), resized_modality_image)
                np.save(os.path.join(new_image_dir,f"{dataset_dir}|{mode}|{save_name}|resize_seg"), resized_modality_label)

            #create the label matrix for resized image
            if len(resized_modality_image.shape) == 2:
                resized_label_amount_matrix = np.zeros([1,int(np.amax(unique_labels))+1])
            else:
                #This can create issues with plan sampling, need to make sure only chosen axes can be sampled from
                resized_label_amount_matrix = np.zeros([3, proc_size, int(np.amax(unique_labels))+1])

            #create the info dictionaries for the original images
            proc_label_dict = {}
            if len(resized_modality_image.shape) == 2:
                proc_label_dict[0] = np.ones([1,int(np.amax(unique_labels))+1])
            else:
                for plane in planes:
                    proc_label_dict[plane] = np.zeros([1,original_normalized_image.shape[plane],int(np.amax(unique_labels))+1])

            #Get rid of zero as a possible label
            unique_labels = unique_labels[1:]

            for plane in planes:
                for label in unique_labels:
                    label = int(label)
                    original_binary_mask = np.int32((label==original_normalized_label))                                       
                    binary_mask = np.int32((label==resized_modality_label))

                    axes = [0,1,2]
                    axes.remove(plane)
                    ax = tuple(axes)

                    if len(original_normalized_label.shape) == 2:
                        num_nonzero_axis = np.count_nonzero(binary_mask)
                        original_num_nonzero_axis = np.count_nonzero(original_binary_mask)

                        resized_label_amount_matrix[0,int(label)] = num_nonzero_axis
                        proc_label_dict[plane][0,int(label)] = original_num_nonzero_axis
                    else:                               
                        num_nonzero_axis = np.count_nonzero(binary_mask, axis=ax)
                        original_num_nonzero_axis = np.count_nonzero(original_binary_mask,axis=ax)

                        resized_label_amount_matrix[plane,:,int(label)] = num_nonzero_axis
                        proc_label_dict[plane][0,:,int(label)] = original_num_nonzero_axis

                if show_imgs:
                    f, axarr = plt.subplots(nrows=1,ncols=4,figsize=[16,4])
                    if len(original_normalized_label.shape) == 2:
                        chosen_original_slice = original_normalized_image
                        chosen_original_label = original_normalized_label

                        chosen_resized_slice = resized_modality_image
                        chosen_resized_label = resized_modality_label
                    else:
                        chosen_original_slice = np.take(original_normalized_image, int(original_normalized_image.shape[plane]/2), plane)
                        chosen_original_label = np.take(original_normalized_label, int(original_normalized_image.shape[plane]/2), plane)

                        chosen_resized_slice = np.take(resized_modality_image, int(resized_modality_image.shape[plane]/2), plane,)
                        chosen_resized_label = np.take(resized_modality_label, int(resized_modality_image.shape[plane]/2), plane)

                    img_obj = axarr[0].imshow(chosen_resized_slice, interpolation="none", cmap="gray")
                    plt.colorbar(img_obj, ax=axarr[0])
                    seg_obj = axarr[1].imshow(chosen_resized_label)
                    plt.colorbar(seg_obj, ax=axarr[1])
                    og_img_obj = axarr[2].imshow(chosen_original_slice, interpolation="none", cmap="gray")
                    plt.colorbar(og_img_obj, ax=axarr[2])
                    og_seg_obj = axarr[3].imshow(chosen_original_label)
                    plt.colorbar(og_seg_obj, ax=axarr[3])
                    plt.show()

            if save_slices:
                save_name = image.split(".")[0]                                        
                new_image_dir = os.path.join(new_mode_dir,save_name)
                dump_dictionary(proc_label_dict, os.path.join(new_image_dir,f"{dataset_dir}|{mode}|{save_name}|original_label_array"))
                np.save(os.path.join(new_image_dir,f"{dataset_dir}|{mode}|{save_name}|resized_label_array"), resized_label_amount_matrix)


    def proc_func(self,
                dset_name,
                dset_info, 
                save_slices=False, 
                show_hists=False,
                show_imgs=False,
                redo_processed=True):

        processed_dir = make_processed_dir(dset_name, dset_info, save_slices)

        image_list = os.listdir(dset_info["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):
                        im_dir = os.path.join(dset_info["image_root_dir"], image, f"ses-BL/anat/{image}_ses-BL_T1w.nii.gz")
                        label_dir = os.path.join(dset_info["label_root_dir"], image, f"ses-FU/anat/{image}_ses-FU_T1w.nii.gz")

                        loaded_image = np.array(nib.load(im_dir).dataobj)
                        loaded_label = np.array(nib.load(label_dir).dataobj)

                        print(loaded_image.shape)
                        print(loaded_label.shape)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        produce_slices(processed_dir,
                                        dset_name,
                                        loaded_image,
                                        loaded_label,
                                        dset_info["modality_names"],
                                        image, 
                                        planes=dset_info["planes"],
                                        proc_size=dset_info["proc_size"],
                                        save_slices=save_slices, 
                                        show_hists=show_hists,
                                        show_imgs=show_imgs,
                                        do_clip=dset_info["do_clip"],
                                        clip_args=dset_info["clip_args"],
                                        norm_scheme=dset_info["norm_scheme"])
                except Exception as e:
                    print(e)
                    raise ValueError
                pbar.update(1)
        pbar.close()