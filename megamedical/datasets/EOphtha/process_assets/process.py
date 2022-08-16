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

#New line!
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class EOphtha:

    def __init__(self):
        self.name = "EOphtha"
        self.dset_info = {
            "e_optha_EX":{
                "main": "EOphtha",
                "image_root_dir":f"{paths['DATA']}/EOphtha/processed/original_unzipped/retreived_2022_03_11/e_optha_EX/EX",
                "label_root_dir":f"{paths['DATA']}/EOphtha/processed/original_unzipped/retreived_2022_03_11/e_optha_EX/Annotation_EX",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            },
             "e_optha_MA":{
                "main": "EOphtha",
                "image_root_dir":f"{paths['DATA']}/EOphtha/processed/original_unzipped/retreived_2022_03_11/e_optha_MA/MA",
                "label_root_dir":f"{paths['DATA']}/EOphtha/processed/original_unzipped/retreived_2022_03_11/e_optha_MA/Annotation_MA",
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
                  version=None,
                show_hists=False,
                  show_imgs=False,
                  save_slices=False,
                redo_processed=True):
        assert not(version is None and save_slices), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = pps.make_processed_dir(self.name, dset_name, save_slices, version)
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):
                        sub_im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        sub_label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image)

                        im_dir = os.path.join(sub_im_dir, os.listdir(sub_im_dir)[0])
                        label_dir = os.path.join(sub_label_dir, os.listdir(sub_label_dir)[0])

                        loaded_image = np.array(Image.open(im_dir).convert('L'))
                        loaded_label = np.array(Image.open(label_dir).convert('L'))

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