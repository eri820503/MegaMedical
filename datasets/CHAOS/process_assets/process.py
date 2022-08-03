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

#New line!
import universeg as uvs
from scripts import preprocess_scripts


class CHAOS:

    def __init__(self):
        
        self.dataset_info_dictionary = {
            "CT":{
                "main":"CHAOS",
                "image_root_dir":"/home/vib9/src/data/CHAOS/processed/original_unzipped/retreived_2022_03_08/Train_Sets/CT",
                "label_root_dir":"/home/vib9/src/data/CHAOS/processed/original_unzipped/retreived_2022_03_08/Train_Sets/CT",
                "modality_names":["CT"],
                "planes":[0],
                "clip_args": [600,1500],
                "norm_scheme": "CT",
                "do_clip": True,
                "proc_size":256
            },
            "MR":{
                "main":"CHAOS",
                "image_root_dir":"/home/vib9/src/data/CHAOS/processed/original_unzipped/retreived_2022_03_08/Train_Sets/MR",
                "label_root_dir":"/home/vib9/src/data/CHAOS/processed/original_unzipped/retreived_2022_03_08/Train_Sets/MR",
                "modality_names":["T2"],
                "planes":[0],
                "clip_args": None,
                "norm_scheme": "MR",
                "do_clip": True,
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
                        if dset_name == "CT":
                            DicomDir = os.path.join(dset_info["image_root_dir"], image, "DICOM_anon")
                            GroundDir = os.path.join(dset_info["image_root_dir"], image, "Ground")
                        else:
                            DicomDir = os.path.join(dset_info["image_root_dir"], image, "T2SPIR/DICOM_anon")
                            GroundDir = os.path.join(dset_info["image_root_dir"], image, "T2SPIR/Ground")

                        planes = []
                        for plane in os.listdir(DicomDir):
                            planes.append(dicom.dcmread(os.path.join(DicomDir, plane)).pixel_array)
                        loaded_image = np.stack(planes)

                        gt_planes = []
                        for gt_plane in os.listdir(GroundDir):
                            gt_planes.append(np.array(Image.open(os.path.join(GroundDir, gt_plane)).convert('L')))
                        loaded_label = np.stack(gt_planes)

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
                    raise ValueError
                pbar.update(1)
        pbar.close()