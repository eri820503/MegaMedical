import nibabel as nib
from tqdm import tqdm
import pydicom as dicom
import plistlib
import numpy as np
import glob
import xml.dom.minidom
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


# Credit goes to https://github.com/msingh9/cs230-Coronary-Calcium-Scoring-/blob/main/code/my_lib.py
def process_xml(f):
    # cornary name to id
    cornary_name_2_id = {"Right Coronary Artery": 0,
                         "Left Anterior Descending Artery": 1,
                         "Left Coronary Artery": 2,
                         "Left Circumflex Artery": 3}
    # input XML file
    # output - directory containing various meta data
    with open(f, 'rb') as fin:
        pl = plistlib.load(fin)
    # extract needed info from XML
    data = {}
    for image in pl["Images"]:
        iidx = image["ImageIndex"]
        for roi in image["ROIs"]:
            print(roi['Point_mm'])
            print(roi['Point_px'])
            print("##########")
            data[iidx][roi]['Point_mm'] = roi['Point_mm']
            data[iidx][roi]['Point_px'] = roi['Point_px']
    return data

class COCA:

    def __init__(self):
        self.name = "COCA"
        self.dset_info = {
            "Challenge2017":{
                "main":"ACDC",
                "image_root_dir":f"{paths['DATA']}/COCA/original_unzipped/retrieved_2022_03_24/Gated_release_final/patient",
                "label_root_dir":f"{paths['DATA']}/COCA/original_unzipped/retrieved_2022_03_24/Gated_release_final/calcium_xml",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            }
        }

    def proc_func(self,
                  dset_name,
                  proc_func,
                  version=None,
                  show_hists=False,
                  show_imgs=False,
                  save_slices=False,
                  redo_processed=True):
        assert not(version is None and save_slices), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = pps.make_processed_dir(self.name, dset_name, save_slices, version, self.dset_info[dset_name])
        image_list = os.listdir(self.dset_info[dset_name]["label_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    image = image.split(".")[0]
                    proc_dir_template = os.path.join(proc_dir, f"megamedical_v{version}", dset_name, "*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                        #flat_img_slices = glob.glob(os.path.join(self.dset_info[dset_name]["image_root_dir"],image,"*/*"))
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],f"{image}.xml")
                        keypoints = process_xml(label_dir)
                        print(keypoints)
                        
                        #loaded_image = np.stack([dicom.dcmread(image_path) for image_path in flat_img_slices])
                        
                        loaded_label = None
                        #loaded_label = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, ".xml")

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        proc_func(proc_dir,
                                  version,
                                  dset_name,
                                  image, 
                                  loaded_image,
                                  loaded_label,
                                  self.dset_info[dset_name],
                                  show_hists=show_hists,
                                  show_imgs=show_imgs,
                                  save_slices=save_slices)
                except Exception as e:
                    print(e)
                    raise ValueError
                pbar.update(1)
        pbar.close()
