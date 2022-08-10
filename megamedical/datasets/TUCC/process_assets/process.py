from ast import DictComp
from multiprocessing.sharedctypes import Value
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
import h5py

#New line!
from megamedical.utils.registry import paths


class TUCC:

    def __init__(self):
        
        self.dataset_info_dictionary = {
            "retreived_2022_03_06":{
                "main": "TUCC",
                "image_root_dir":"/home/vib9/src/data/TUCC/processed/original_unzipped/retreived_2022_03_04",
                "label_root_dir":"/home/vib9/src/data/TUCC/processed/original_unzipped/retreived_2022_03_04",
                "modality_names":["NA"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":False,
                "proc_size":256
            }
        }

    def proc_func(self,
                dset_name,
                processed_dir,
                redo_processed=True):
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        image_set = []
        seg_set = []
        hf = h5py.File(os.path.join(self.dset_info[dset_name]["image_root_dir"],'dataset.hdf5'), 'r')
        images = np.array(hf["image"][:1000])
        segs = np.array(hf["mask"][:1000])

        with tqdm(total=1000, desc=f'Processing: {dset_name}', unit='image') as pbar:
            for idx, image in enumerate([f"img{i}" for i in range(1000)]):
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):

                        loaded_image = images[idx, ...]
                        loaded_label = segs[idx, ...]

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        image_set.append(loaded_image)
                        seg_set.append(loaded_label)
                except Exception as e:
                    print(e)
                pbar.update(1)
        pbar.close()
        return image_set, seg_set