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


class OCTA500:

    def __init__(self):
        
        self.dataset_info_dictionary = {
            "OCTA_3M":{
                "main":"OCTA500",
                "image_root_dir":"/home/vib9/src/data/OCTA500/processed/original_unzipped/retrieved_04_01/OCTA_3M/Projection Maps/OCT(FULL)",
                "label_root_dir":"/home/vib9/src/data/OCTA500/processed/original_unzipped/retrieved_04_01/OCTA_3M/GroundTruth",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            },
            "OCTA_6M":{
                "main":"OCTA500",
                "image_root_dir":"/home/vib9/src/data/OCTA500/processed/original_unzipped/retrieved_04_01/OCTA_6M/Projection Maps/OCT(FULL)",
                "label_root_dir":"/home/vib9/src/data/OCTA500/processed/original_unzipped/retrieved_04_01/OCTA_6M/GroundTruth",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
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
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image)

                        loaded_image = io.imread(im_dir)
                        loaded_label = io.imread(label_dir)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        images.append(loaded_image)
                        segs.append(loaded_label)
                except Exception as e:
                    print(e)
                pbar.update(1)
        pbar.close()
        return images, segs