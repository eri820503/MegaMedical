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
    "ISLES2017":{
        "main":"ISLES",
        "image_root_dir":"/home/vib9/src/data/ISLES/processed/original_unzipped/ISLES2017/training",
        "label_root_dir":"/home/vib9/src/data/ISLES/processed/original_unzipped/ISLES2017/training",
        "modality_names":["ADC","MIT","TTP","Tmax","rCBF","rCBV"],
        "planes":[2],
        "clip_args":None,
        "norm_scheme":"MR",
        "do_clip":True,
        "proc_size":256
    }
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
                    
                    ADC_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_ADC*/VSD.Brain.XX.O.MR_ADC*.nii"))[0]
                    MIT_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_MTT*/VSD.Brain.XX.O.MR_MTT*.nii"))[0]
                    TTP_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_TTP*/VSD.Brain.XX.O.MR_TTP*.nii"))[0]
                    Tmax_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_Tmax*/VSD.Brain.XX.O.MR_Tmax*.nii"))[0]
                    rCBF_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_rCBF*/VSD.Brain.XX.O.MR_rCBF*.nii"))[0]
                    rCBV_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_rCBV*/VSD.Brain.XX.O.MR_rCBV*.nii"))[0]

                    label_dir = glob.glob(os.path.join(dset_info["label_root_dir"], image, "VSD.Brain.XX.O.OT*/VSD.Brain.XX.O.OT*.nii"))[0]
                    
                    ADC = np.array(nib.load(ADC_im_dir).dataobj)
                    MIT = np.array(nib.load(MIT_im_dir).dataobj)
                    TTP = np.array(nib.load(TTP_im_dir).dataobj)
                    Tmax = np.array(nib.load(Tmax_im_dir).dataobj)
                    rCBF = np.array(nib.load(rCBF_im_dir).dataobj)
                    rCBV = np.array(nib.load(rCBV_im_dir).dataobj)

                    loaded_image = np.stack([ADC, MIT, TTP, Tmax, rCBF, rCBV], -1)
                    loaded_label = np.array(nib.load(label_dir).dataobj)

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
    dtp = ["ISLES2017"]
    process_datasets(dtp=dtp, save_volumes=True, show_imgs=False, show_hists=False, redo_processed=True)
