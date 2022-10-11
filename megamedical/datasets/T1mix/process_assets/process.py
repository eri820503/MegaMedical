import nibabel as nib
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

#New line!
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
                  dset_name,
                  proc_func,
                  load_images=True,
                  accumulate=False,
                  version=None,
                  show_imgs=False,
                  save=False,
                  show_hists=False,
                  resolutions=None,
                  redo_processed=True):
        assert not(version is None and save), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        image_list = sorted(os.listdir(self.dset_info[dset_name]["image_root_dir"]))
        proc_dir = os.path.join(paths['ROOT'], "processed")
        res_dict = {}
        subj_dict = {}
        for resolution in resolutions:
            accumulator = []
            subj_accumulator = []
            for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
                try:
                    # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
                    template_root = os.path.join(proc_dir, f"res{resolution}", self.name)
                    mid_proc_dir_template = os.path.join(template_root, f"midslice_v{version}", dset_name, "*/*", image)
                    max_proc_dir_template = os.path.join(template_root, f"maxslice_v{version}", dset_name, "*/*", image)
                    if redo_processed or (len(glob.glob(mid_proc_dir_template)) == 0) or (len(glob.glob(max_proc_dir_template)) == 0):
                        # Skull-stripped
                        vol_im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        # Original volume
                        norm_im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"].replace("vols", "origs"), image.replace("norm", "orig"))
                        # Segmentation
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image.replace("norm", "aseg"))
                        
                        if load_images:
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
                        subj_name = image.split(".")[0]
                        proc_return = proc_func(proc_dir,
                                                version,
                                                dset_name,
                                                subj_name, 
                                                loaded_image,
                                                loaded_label,
                                                self.dset_info[dset_name],
                                                show_hists=show_hists,
                                                show_imgs=show_imgs,
                                                res=resolution,
                                                save=save)
                        
                        if accumulate:
                            accumulator.append(proc_return)
                            subj_accumulator.append(subj_name)
                except Exception as e:
                    print(e)
                    #raise ValueError
            res_dict[resolution] = accumulator
            subj_dict[resolution] = subj_accumulator
        if accumulate:
            return proc_dir, subj_dict, res_dict