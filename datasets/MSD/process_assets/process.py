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

dataset_info_dictionary = {
    "BrainTumour":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/BrainTumour/retrieved_2018_07_03/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/BrainTumour/retrieved_2018_07_03/segs",
        "modality_names":["FLAIR", "T1w", "T1gd","T2w"],
        "planes":[0, 1, 2],
        "clip_args":None,
        "norm_scheme":"MR",
        "do_clip":True,
        "proc_size":256
    },
    "Heart":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Heart/retrieved_2021_04_25/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Heart/retrieved_2021_04_25/segs",
        "modality_names":["Mono"],
        "planes":[0, 1, 2],
        "clip_args":None,
        "norm_scheme":"MR",
        "do_clip":True,
        "proc_size":256
    },
    "Liver":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Liver/retrieved_2018_05_26/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Liver/retrieved_2018_05_26/segs",
        "modality_names":["PVP-CT"],
        "planes":[0, 1, 2],
        "clip_args":[-250,250],
        "norm_scheme":"CT",
        "do_clip":True,
        "proc_size":256
    },
    "Hippocampus":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Hippocampus/retrieved_2021_04_22/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Hippocampus/retrieved_2021_04_22/segs",
        "modality_names":["Mono"],
        "planes":[0, 1, 2],
        "clip_args":None,
        "norm_scheme":"MR",
        "do_clip":True,
        "proc_size":256
    },
    "Prostate":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Prostate/retrieved_2018_05_31/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Prostate/retrieved_2018_05_31/segs",
        "modality_names":["T2","ADC"],
        "planes":[0, 1, 2],
        "clip_args":None,
        "norm_scheme":"MR",
        "do_clip":True,
        "proc_size":256
    },
    "Lung":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Lung/retrieved_2018_05_31/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Lung/retrieved_2018_05_31/segs",
        "modality_names":["CT"],
        "planes":[0, 1, 2],
        "clip_args":[-500,1000],
        "norm_scheme":"CT",
        "do_clip":True,
        "proc_size":256
    },
    "Pancreas":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Pancreas/retrieved_2021_04_22/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Pancreas/retrieved_2021_04_22/segs",
        "modality_names":["PVP-CT"],
        "planes":[0, 1, 2],
        "clip_args":[-500,1000],
        "norm_scheme":"CT",
        "do_clip":True,
        "proc_size":256
    },
    "HepaticVessel":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/HepaticVessel/retrieved_2021_04_22/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/HepaticVessel/retrieved_2021_04_22/segs",
        "modality_names":["CT"],
        "planes":[0, 1, 2],
        "clip_args":[-500,1000],
        "norm_scheme":"CT",
        "do_clip":True,
        "proc_size":256
    },
    "Spleen":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Spleen/retrieved_2021_04_22/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Spleen/retrieved_2021_04_22/segs",
        "modality_names":["CT"],
        "planes":[0, 1, 2],
        "clip_args":[-500,1000],
        "norm_scheme":"CT",
        "do_clip":True,
        "proc_size":256
    },
    "Colon":{
        "main":"MSD",
        "image_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Colon/retrieved_2021_04_22/images",
        "label_root_dir":"/home/vib9/src/data/MSD/processed/original_unzipped/Colon/retrieved_2021_04_22/segs",
        "modality_names":["CT"],
        "planes":[0, 1, 2],
        "clip_args":[-500,1000],
        "norm_scheme":"CT",
        "do_clip":True,
        "proc_size":256
    },
}


def proc_func(dset_name,
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
                    im_dir = os.path.join(dset_info["image_root_dir"], image)
                    label_dir = os.path.join(dset_info["label_root_dir"], image)

                    assert os.path.isfile(im_dir), "Valid image dir required!"
                    assert os.path.isfile(label_dir), "Valid label dir required!"

                    loaded_image = preprocess_scripts.resample_nib(nib.load(im_dir))
                    loaded_label = preprocess_scripts.resample_mask_to(nib.load(label_dir), loaded_image)
                    #loaded_image = nib.load(im_dir)
                    #loaded_label = nib.load(label_dir)

                    loaded_image = np.array(loaded_image.dataobj)
                    loaded_label = np.array(loaded_label.dataobj)

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
                
            pbar.update(1)
    pbar.close()

def process_datasets(dtp,
                    save_volumes, 
                    show_imgs, 
                    show_hists,
                    redo_processed):
    for subset in dtp:
        proc_func(
                dset_name=subset,
                dset_info=dataset_info_dictionary[subset],
                save_slices=save_volumes,
                show_hists=show_hists,
                show_imgs=show_imgs,
                redo_processed=redo_processed
        )

if __name__ == "__main__":
    #Colon, HepaticVessel, Liver
    dtp = ["Heart", "Liver"]
    process_datasets(dtp=dtp, save_volumes=True, show_imgs=False, show_hists=False, redo_processed=True)
