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

#New line!
import universeg as uvs
from scripts import preprocess_scripts


dataset_info_dictionary = {
    "2016_LGG":{
        "main":"BRATS",
        "image_root_dir":"/home/vib9/src/data/BRATS/processed/original_unzipped/2016/BRATS2015_Training/LGG",
        "label_root_dir":"/home/vib9/src/data/BRATS/processed/original_unzipped/2016/BRATS2015_Training/LGG",
        "modality_names":["FLAIR","T1","T1c","T2"],
        "planes":[0, 1, 2],
        "clip_args":None,
        "norm_scheme":"MR",
        "do_clip":True,
        "proc_size":256
    },
    "2016_HGG":{
        "main":"BRATS",
        "image_root_dir":"/home/vib9/src/data/BRATS/processed/original_unzipped/2016/BRATS2015_Training/HGG",
        "label_root_dir":"/home/vib9/src/data/BRATS/processed/original_unzipped/2016/BRATS2015_Training/HGG",
        "modality_names":["FLAIR","T1","T1c","T2"],
        "planes":[0, 1, 2],
        "clip_args":None,
        "norm_scheme":"MR",
        "do_clip":True,
        "proc_size":256
    },
    "2021":{
        "main":"BRATS",
        "image_root_dir":"/home/vib9/src/data/BRATS/processed/original_unzipped/2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021",
        "label_root_dir":"/home/vib9/src/data/BRATS/processed/original_unzipped/2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021",
        "modality_names":["FLAIR","T1","T1c","T2"],
        "planes":[0, 1, 2],
        "clip_args":None,
        "norm_scheme":"MR",
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
                    subj_folder = os.path.join(dset_info["image_root_dir"], image)
                    if dset_name == "2021":
                        flair_im_dir = os.path.join(subj_folder, f"{image}_flair.nii.gz")
                        t1_im_dir = os.path.join(subj_folder, f"{image}_t1.nii.gz")
                        t1c_im_dir = os.path.join(subj_folder, f"{image}_t1ce.nii.gz")
                        t2_im_dir = os.path.join(subj_folder, f"{image}_t2.nii.gz")

                        label_dir = os.path.join(dset_info["label_root_dir"], image, f"{image}_seg.nii.gz")

                        flair_image = np.array(nib.load(flair_im_dir).dataobj)
                        t1_image = np.array(nib.load(t1_im_dir).dataobj)
                        t1c_image = np.array(nib.load(t1c_im_dir).dataobj)
                        t2_image = np.array(nib.load(t2_im_dir).dataobj)

                        loaded_image = np.stack([flair_image, t1_image, t1c_image, t2_image])
                        loaded_label = np.array(nib.load(label_dir).dataobj)
                    else:
                        flair_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_Flair*/VSD.Brain.XX.O.MR_Flair*.mha"))[0]
                        t1_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T1*/VSD.Brain.XX.O.MR_T1*.mha"))[0]
                        t1c_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T1c*/VSD.Brain.XX.O.MR_T1c*.mha"))[0]
                        t2_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T2*/VSD.Brain.XX.O.MR_T2*.mha"))[0]

                        label_dir = glob.glob(os.path.join(dset_info["label_root_dir"], image, "VSD.Brain_*/VSD.Brain_*.mha"))[0]

                        flair_image, _ = medpy.io.load(flair_im_dir)
                        t1_image, _ = medpy.io.load(t1_im_dir)
                        t1c_image, _ = medpy.io.load(t1c_im_dir)
                        t2_image, _ = medpy.io.load(t2_im_dir)

                        loaded_image = np.stack([flair_image, t1_image, t1c_image, t2_image], -1)
                        loaded_label, _ = medpy.io.load(label_dir)

                    assert not (loaded_image is None), "Invalid Image"
                    assert not (loaded_label is None), "Invalid Label"

                    print(loaded_image.shape)
                    print(loaded_label.shape)

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
    dtp = ["2016_LGG", "2016_HGG", "2021"]
    process_datasets(dtp=dtp, save_volumes=True, show_imgs=False, show_hists=False, redo_processed=True)
