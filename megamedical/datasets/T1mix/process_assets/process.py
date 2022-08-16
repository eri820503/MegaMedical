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
from megamedical.utils import proc_utils as put


class T1mix:

    def __init__(self):
        self.name = "T1mix"
        self.dset_info = {
            "retrieved_2021_06_10":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/processed/original_unzipped/retrieved_2021_06_10/train/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/processed/original_unzipped/retrieved_2021_06_10/train/asegs",
                "modality_names":["T1"],
                "planes":[0, 1, 2],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
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
                    if not("OASIS" in image) and (redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0)):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image.replace("norm", "aseg"))

                        loaded_image = np.load(im_dir)['vol_data']
                        loaded_label = np.load(label_dir)['vol_data']

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        #New line!
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