import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import numpy as np
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class VerSe:

    def __init__(self):
        self.name = "VerSe"
        self.dset_info = {
            "VerSe19":{
                "main": "VerSe",
                "image_root_dir":f"{paths['DATA']}/VerSe/original_unzipped/VerSe19/dataset-verse19training/rawdata",
                "label_root_dir":f"{paths['DATA']}/VerSe/original_unzipped/VerSe19/dataset-verse19training/derivatives",
                "modality_names":["CT"],
                "planes": [0],
                "clip_args":[-500,1000],
                "norm_scheme":"CT"
            },
            "VerSe20":{
                "main": "VerSe",
                "image_root_dir":f"{paths['DATA']}/VerSe/original_unzipped/VerSe20/dataset-01training/rawdata",
                "label_root_dir":f"{paths['DATA']}/VerSe/original_unzipped/VerSe20/dataset-01training/derivatives",
                "modality_names":["CT"],
                "planes": [0, 1],
                "clip_args":[-500,1000],
                "norm_scheme":"CT"
            }
        }

    def proc_func(self,
                  subdset,
                  pps_function,
                  parallelize=False,
                  load_images=True,
                  accumulate=False,
                  version=None,
                  show_imgs=False,
                  save=False,
                  show_hists=False,
                  resolutions=None,
                  redo_processed=True):
        assert not(version is None and save), "Must specify version for saving."
        assert subdset in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = os.path.join(paths['ROOT'], "processed")
        image_list = sorted(os.listdir(self.dset_info[subdset]["image_root_dir"]))
        subj_dict, res_dict = proc.process_image_list(process_VerSe_image,
                                                      proc_dir,
                                                      image_list,
                                                      parallelize,
                                                      pps_function,
                                                      resolutions,
                                                      self.name,
                                                      subdset,
                                                      self.dset_info,
                                                      redo_processed,
                                                      load_images,
                                                      show_hists,
                                                      version,
                                                      show_imgs,
                                                      accumulate,
                                                      save)
        if accumulate:
            return proc_dir, subj_dict, res_dict
        
        
global process_VerSe_image
def process_VerSe_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed'] or is_processed_check(item):
            if item['subdset'] == "VerSe19":
                im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'], f"{item['image']}_ct.nii.gz")
                label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'], f"{item['image']}_seg-vert_msk.nii.gz")
            else:
                im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'], f"{item['image']}_dir-ax_ct.nii.gz")
                label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'], f"{item['image']}_dir-ax_seg-vert_msk.nii.gz")

            if item['load_images']:
                loaded_image = put.resample_nib(nib.load(im_dir))
                loaded_label = put.resample_mask_to(nib.load(label_dir), loaded_image)

                loaded_image = loaded_image.get_fdata()
                loaded_label = loaded_label.get_fdata()
                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = nib.load(label_dir).get_fdata()

            # Set the name to be saved
            subj_name = item['image'].split(".")[0]
            pps_function = item['pps_function']
            proc_return = pps_function(item['proc_dir'],
                                        item['version'],
                                        item['subdset'],
                                        subj_name, 
                                        loaded_image,
                                        loaded_label,
                                        dset_info[item['subdset']],
                                        show_hists=item['show_hists'],
                                        show_imgs=item['show_imgs'],
                                        res=item['resolution'],
                                        save=item['save'])

            return proc_return, subj_name
        else:
            return None, None
    except Exception as e:
        print(e)
        return None, None