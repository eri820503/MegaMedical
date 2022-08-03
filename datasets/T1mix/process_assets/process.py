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
import universeg as uvs
from scripts import preprocess_scripts


class T1mix:

    def __init__(self):
        
        self.dataset_info_dictionary = {
            "retrieved_2021_06_10":{
                "main":"T1mix",
                "image_root_dir":"/home/vib9/src/data/T1mix/processed/original_unzipped/retrieved_2021_06_10/train/vols",
                "label_root_dir":"/home/vib9/src/data/T1mix/processed/original_unzipped/retrieved_2021_06_10/train/asegs",
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
                    if not("OASIS" in image) and (redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0)):
                        im_dir = os.path.join(dset_info["image_root_dir"], image)
                        label_dir = os.path.join(dset_info["label_root_dir"], image.replace("norm", "aseg"))

                        loaded_image = np.load(im_dir)['vol_data']
                        loaded_label = np.load(label_dir)['vol_data']

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        #New line!
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