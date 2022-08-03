from ast import DictComp
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
import pydicom as dicom
import scipy.io
import nibabel.processing as nip

#New line!
import universeg as uvs
from scripts import preprocess_scripts


class VerSe:

    def __init__(self):

        self.dataset_info_dictionary = {
            "VerSe19":{
                "main": "VerSe",
                "image_root_dir":"/home/vib9/src/data/VerSe/processed/original_unzipped/VerSe19/dataset-verse19training/rawdata",
                "label_root_dir":"/home/vib9/src/data/VerSe/processed/original_unzipped/VerSe19/dataset-verse19training/derivatives",
                "modality_names":["CT"],
                "planes": [0],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            },
            "VerSe20":{
                "main": "VerSe",
                "image_root_dir":"/home/vib9/src/data/VerSe/processed/original_unzipped/VerSe20/dataset-01training/rawdata",
                "label_root_dir":"/home/vib9/src/data/VerSe/processed/original_unzipped/VerSe20/dataset-01training/derivatives",
                "modality_names":["CT"],
                "planes": [0, 1],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            }
        }

    def proc_func(self,
                dset_name,
                dset_info, 
                save_slices=False, 
                show_hists=False,
                show_imgs=False,
                redo_processed=True):

        processed_dir = preprocess_scripts.make_processed_dir(dset_name, dset_info, save_slices)

        image_list = os.listdir(dset_info["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):

                        if dset_name == "VerSe19":
                            im_dir = os.path.join(dset_info["image_root_dir"], image, f"{image}_ct.nii.gz")
                            label_dir = os.path.join(dset_info["label_root_dir"], image, f"{image}_seg-vert_msk.nii.gz")

                            loaded_image = np.array(nib.load(im_dir).dataobj)
                            loaded_label = np.array(nib.load(label_dir).dataobj)
                        else:
                            im_dir = os.path.join(dset_info["image_root_dir"], image, f"{image}_dir-ax_ct.nii.gz")
                            label_dir = os.path.join(dset_info["label_root_dir"], image, f"{image}_dir-ax_seg-vert_msk.nii.gz")

                            loaded_image = resample_nib(nib.load(im_dir))
                            loaded_label = np.array(resample_mask_to(nib.load(label_dir), loaded_image).dataobj)
                            loaded_image = np.array(loaded_image.dataobj)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        preprocess_scripts.produce_slices(processed_dir,
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
                pbar.update(1)
        pbar.close()