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


class LGGFlair:

    def __init__(self):
        self.name = "LGGFlair"
        self.dset_info = {
            "retrieved_2021_10_11":{
                "main":"LGGFlair",
                "image_root_dir":f"{paths['ROOT']}/megamedical/datasets/LGGFlair/processed/original_unzipped/retrieved_2021_10_11",
                "label_root_dir":f"{paths['ROOT']}/megamedical/datasets/LGGFlair/processed/original_unzipped/retrieved_2021_10_11",
                "modality_names":["Flair"],
                "planes":[2],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            }
        }

    def proc_func(self,
                dset_name,
                show_hists=False,
                  show_imgs=False,
                  save_slices=False,
                redo_processed=True):
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = pps.make_processed_dir(dset_name, self.dset_info[dset_name], save_slices)
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):
                        file_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        image_file_list = []
                        label_file_list = []

                        num_slices = int(len(os.listdir(file_dir))/2)
                        for i in range(1, num_slices + 1):
                            image_file_list.append(os.path.join(file_dir,f"{image}_{str(i)}.tif"))
                            label_file_list.append(os.path.join(file_dir,f"{image}_{str(i)}_mask.tif"))

                        loaded_image = np.concatenate([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)[...,np.newaxis] for f in image_file_list], axis=2)
                        loaded_label = np.concatenate([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)[...,np.newaxis] for f in label_file_list], axis=2)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        pps.produce_slices(proc_dir,
                                          dset_name,
                                          loaded_image,
                                          loaded_label,
                                          self.dset_info[dset_name],
                                          show_hists=show_hists,
                                          show_imgs=show_imgs)
                except Exception as e:
                    print(e)
                pbar.update(1)
        pbar.close()