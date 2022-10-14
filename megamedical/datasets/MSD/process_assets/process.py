import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import numpy as np
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class MSD:

    def __init__(self):
        self.name = "MSD"
        self.dset_info = {
            "BrainTumour":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/BrainTumour/retrieved_2018_07_03/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/BrainTumour/retrieved_2018_07_03/segs",
                "modality_names":["FLAIR", "T1w", "T1gd","T2w"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "Heart":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Heart/retrieved_2021_04_25/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Heart/retrieved_2021_04_25/segs",
                "modality_names":["Mono"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "Liver":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Liver/retrieved_2018_05_26/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Liver/retrieved_2018_05_26/segs",
                "modality_names":["PVP-CT"],
                "planes":[0, 1, 2],
                "clip_args":[-250,250],
                "norm_scheme":"CT"
            },
            "Hippocampus":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Hippocampus/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Hippocampus/retrieved_2021_04_22/segs",
                "modality_names":["Mono"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "Prostate":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Prostate/retrieved_2018_05_31/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Prostate/retrieved_2018_05_31/segs",
                "modality_names":["T2","ADC"],
                "planes":[2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "Lung":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Lung/retrieved_2018_05_31/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Lung/retrieved_2018_05_31/segs",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT"
            },
            "Pancreas":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Pancreas/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Pancreas/retrieved_2021_04_22/segs",
                "modality_names":["PVP-CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT"
            },
            "HepaticVessel":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/HepaticVessel/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/HepaticVessel/retrieved_2021_04_22/segs",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT"
            },
            "Spleen":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Spleen/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Spleen/retrieved_2021_04_22/segs",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT"
            },
            "Colon":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Colon/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Colon/retrieved_2021_04_22/segs",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT"
            },
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
        subj_dict, res_dict = proc.process_image_list(process_MSD_image,
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

        
global process_MSD_image
def process_MSD_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed'] or is_processed_check(item):
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'])

            assert os.path.isfile(im_dir), "Valid image dir required!"
            assert os.path.isfile(label_dir), "Valid label dir required!"

            if item['load_images']:
                loaded_image = nib.load(im_dir)
                loaded_label = nib.load(label_dir)

                if len(loaded_image.shape) == 3:
                    loaded_image = put.resample_nib(loaded_image)
                    loaded_label = put.resample_mask_to(loaded_label, loaded_image)

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