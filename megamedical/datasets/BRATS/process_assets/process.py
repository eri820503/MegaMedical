import medpy.io
from tqdm.notebook import tqdm_notebook
import nibabel as nib
import numpy as np
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class BRATS:

    def __init__(self):
        self.name = "BRATS"
        self.dset_info = {
            "2016_LGG":{
                "main":"BRATS",
                "image_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2016/BRATS2015_Training/LGG",
                "label_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2016/BRATS2015_Training/LGG",
                "modality_names":["FLAIR","T1","T1c","T2"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "2016_HGG":{
                "main":"BRATS",
                "image_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2016/BRATS2015_Training/HGG",
                "label_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2016/BRATS2015_Training/HGG",
                "modality_names":["FLAIR","T1","T1c","T2"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "2021":{
                "main":"BRATS",
                "image_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021",
                "label_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021",
                "modality_names":["FLAIR","T1","T1c","T2"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
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
        subj_dict, res_dict = proc.process_image_list(process_BRATS_image,
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
        
        
global process_BRATS_image
def process_BRATS_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed'] or is_processed_check(item):
            subj_folder = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            if item['subdset'] == "2021":
                if item['load_images']:
                    flair_im_dir = os.path.join(subj_folder, f"{item['image']}_flair.nii.gz")
                    t1_im_dir = os.path.join(subj_folder, f"{item['image']}_t1.nii.gz")
                    t1c_im_dir = os.path.join(subj_folder, f"{item['image']}_t1ce.nii.gz")
                    t2_im_dir = os.path.join(subj_folder, f"{item['image']}_t2.nii.gz")
                    label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'], f"{item['image']}_seg.nii.gz")

                    flair_image = nib.load(flair_im_dir).get_fdata()
                    t1_image = nib.load(t1_im_dir).get_fdata()
                    t1c_image = nib.load(t1c_im_dir).get_fdata()
                    t2_image = nib.load(t2_im_dir).get_fdata()

                    loaded_image = np.stack([flair_image, t1_image, t1c_image, t2_image], axis=-1)
                    loaded_label = nib.load(label_dir).get_fdata()
                    assert not (loaded_label is None), "Invalid Label"
                    assert not (loaded_image is None), "Invalid Image"
                else:
                    label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'], f"{item['image']}_seg.nii.gz")
                    loaded_image = None
                    loaded_label = nib.load(label_dir).get_fdata()
            else:
                if item['load_images']:
                    flair_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_Flair*/VSD.Brain.XX.O.MR_Flair*.mha"))[0]
                    t1_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T1*/VSD.Brain.XX.O.MR_T1*.mha"))[0]
                    t1c_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T1c*/VSD.Brain.XX.O.MR_T1c*.mha"))[0]
                    t2_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T2*/VSD.Brain.XX.O.MR_T2*.mha"))[0]
                    label_dir = glob.glob(os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'], "VSD.Brain_*/VSD.Brain_*.mha"))[0]

                    flair_image, _ = medpy.io.load(flair_im_dir)
                    t1_image, _ = medpy.io.load(t1_im_dir)
                    t1c_image, _ = medpy.io.load(t1c_im_dir)
                    t2_image, _ = medpy.io.load(t2_im_dir)

                    loaded_image = np.stack([flair_image, t1_image, t1c_image, t2_image], axis=-1)
                    loaded_label, _ = medpy.io.load(label_dir)
                    assert not (loaded_label is None), "Invalid Label"
                    assert not (loaded_image is None), "Invalid Image"
                else:
                    label_dir = glob.glob(os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'], "VSD.Brain_*/VSD.Brain_*.mha"))[0]

                    loaded_image = None
                    loaded_label, _ = medpy.io.load(label_dir)

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