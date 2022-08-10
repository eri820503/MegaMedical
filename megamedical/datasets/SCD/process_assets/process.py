from multiprocessing.sharedctypes import Value
from operator import truediv
from select import select
from turtle import pos
import numpy as np
from torch import mul
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

#New line!
from megamedical.utils.registry import paths


class SCD:

    def __init__(self):

        dataset_info_dictionary = {
            "LAS":{
                "main":"SCD",
                "image_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/LAS/retrieved_2021_11_08",
                "label_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/LAS/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "LAF_Pre":{
                "main":"SCD",
                "image_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/LAF_Pre/retrieved_2021_11_08",
                "label_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/LAF_Pre/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "LAF_Post":{
                "main":"SCD",
                "image_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/LAF_Post/retrieved_2021_11_08",
                "label_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/LAF_Post/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "VIS_pig":{
                "main":"SCD",
                "image_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/VIS_pig/retrieved_2021_11_08",
                "label_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/VIS_pig/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "VIS_human":{
                "main":"SCD",
                "image_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/VIS_human/retrieved_2021_11_08",
                "label_root_dir":"/home/vib9/src/data/SCD/processed/original_unzipped/VIS_human/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            }
        }

    def proc_func(self,
                dset_name,
                processed_dir,
                redo_processed=True):
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        images = []
        segs = []
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):
                        if dset_name=="LAS":
                            im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"],image) + "/image.mhd"
                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],image) + "/gt_binary.mhd"
                            loaded_image = io.imread(im_dir, plugin='simpleitk')
                            loaded_label = io.imread(label_dir, plugin='simpleitk')
                        else:
                            if dset_name=="LAF_Pre":
                                im_dir = os.path.join(self.dset_info["image_root_dir"],image) + f"/de_a_{image[1:]}.nrrd"
                                label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],image) + f"/la_seg_a_{image[1:]}.nrrd"
                            elif dset_name=="LAF_Post":
                                im_dir = os.path.join(self.dset_info["image_root_dir"],image) + f"/de_b_{image[1:]}.nrrd"
                                label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],image) + f"/la_seg_b_{image[1:]}.nrrd"
                            elif dset_name in ["VIS_pig","VIS_human"]:
                                im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"],image) + "/" + image + "_de.nrrd"
                                label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],image) + "/" + image + "_myo.nrrd"
                            loaded_image, _ = nrrd.read(im_dir)
                            loaded_label, _ = nrrd.read(label_dir)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        images.append(loaded_image)
                        segs.append(loaded_label)
                except Exception as e:
                    print(e)
                pbar.update(1)
        pbar.close()
        return images, segs