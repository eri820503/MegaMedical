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
import rawpy
import nibabel as nib
import nibabel.processing as nip


#New line!
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class WORD:

    def __init__(self):
        self.name = "WORD"
        self.dset_info = {
            "retrieved_2022_05_01":{
                "main":"WORD",
                "image_root_dir":"/share/sablab/nfs02/users/gid-dalcaav/data/originals/WORD/processed/unzipped/WORD-V0.1.0-remapped-subset/images",
                "label_root_dir":"/share/sablab/nfs02/users/gid-dalcaav/data/originals/WORD/processed/unzipped/WORD-V0.1.0-remapped-subset/labels",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
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
        image_list = os.listdir(self.dset_info["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for sub_num, image in enumerate(image_list):
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):
                        im_dir = os.path.join(self.dset_info["image_root_dir"], image)
                        label_dir = os.path.join(self.dset_info["label_root_dir"], image)

                        assert os.path.isfile(im_dir), "Valid image dir required!"
                        assert os.path.isfile(label_dir), "Valid label dir required!"

                        loaded_image = put.resample_nib(nib.load(im_dir))
                        loaded_label = put.resample_mask_to(nib.load(label_dir), loaded_image)

                        loaded_image = loaded_image.get_fdata()
                        loaded_label = loaded_label.get_fdata()

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        image_name = f"subj_{sub_num}"

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