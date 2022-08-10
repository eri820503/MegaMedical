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
import universeg as uvs
from scripts import preprocess_scripts


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
                dset_info, 
                save_slices=False, 
                show_hists=False,
                show_imgs=False,
                redo_processed=True):

        processed_dir = preprocess_scripts.make_processed_dir(dset_name, dset_info, save_slices)

        hf = h5py.File(os.path.join(dset_info["image_root_dir"],'dataset.hdf5'), 'r')
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