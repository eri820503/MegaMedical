import nibabel as nib
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class T1mix:

    def __init__(self):
        self.name = "T1mix"
        self.dset_info = {
            # Note OASIS is available in T1mix, but left out on purpose.
            "ABIDE":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/ABIDE/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/ABIDE/asegs",
                "modality_names":["T1", "skull-stripped-T1"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "ADHD200":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/ADHD200/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/ADHD200/asegs",
                "modality_names":["T1", "skull-stripped-T1"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "ADNI":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/ADNI/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/ADNI/asegs",
                "modality_names":["T1", "skull-stripped-T1"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "COBRE":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/COBRE/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/COBRE/asegs",
                "modality_names":["T1", "skull-stripped-T1"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "GSP":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/GSP/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/GSP/asegs",
                "modality_names":["T1", "skull-stripped-T1"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "MCIC":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/MCIC/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/MCIC/asegs",
                "modality_names":["T1", "skull-stripped-T1"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "PPMI":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/PPMI/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/PPMI/asegs",
                "modality_names":["T1", "skull-stripped-T1"],
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
        subj_dict, res_dict = proc.process_image_list(process_T1mix_image,
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

        
global process_T1mix_image
def process_T1mix_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed'] or put.is_processed_check(item):
            # Skull-stripped
            vol_im_dir = os.path.join(dset_info[item['subdset']]['image_root_dir'], item['image'])
            # Original volume
            norm_im_dir = os.path.join(dset_info[item['subdset']]['image_root_dir'].replace("vols", "origs"), item['image'].replace("norm", "orig"))
            # Segmentation
            label_dir = os.path.join(dset_info[item['subdset']]['label_root_dir'], item['image'].replace("norm", "aseg"))

            if item['load_images']:
                loaded_vol_image = np.load(vol_im_dir)['vol_data']
                loaded_norm_image = np.load(norm_im_dir)['vol_data']
                loaded_image = np.stack([loaded_vol_image, loaded_norm_image], -1)
                loaded_label = np.load(label_dir)['vol_data']

                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = np.load(label_dir)['vol_data']

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