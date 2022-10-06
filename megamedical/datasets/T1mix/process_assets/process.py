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
            "retrieved_2021_06_10":{
                "main":"T1mix",
                "image_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/train/vols",
                "label_root_dir":f"{paths['DATA']}/T1mix/original_unzipped/retrieved_2021_06_10/train/asegs",
                "modality_names":["T1"],
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
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        proc_dir = os.path.join(paths['ROOT'], "processed")
        res_dict = {}
        for resolution in resolutions:
            accumulator = []
            for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
                try:
                    # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
                    template_root = os.path.join(proc_dir, f"res{resolution}", self.name)
                    mid_proc_dir_template = os.path.join(template_root, f"midslice_v{version}", dset_name, "*/*", image)
                    max_proc_dir_template = os.path.join(template_root, f"maxslice_v{version}", dset_name, "*/*", image)
                    if not("OASIS" in image) and (redo_processed or (len(glob.glob(mid_proc_dir_template)) == 0) or (len(glob.glob(max_proc_dir_template)) == 0)):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image.replace("norm", "aseg"))

                        if load_images:
                            loaded_image = np.load(im_dir)['vol_data']
                            loaded_label = np.load(label_dir)['vol_data']
                            assert not (loaded_label is None), "Invalid Label"
                            assert not (loaded_image is None), "Invalid Image"
                        else:
                            loaded_image = None
                            loaded_label = np.load(label_dir)['vol_data']

                        proc_return = proc_func(proc_dir,
                                                  version,
                                                  dset_name,
                                                  image, 
                                                  loaded_image,
                                                  loaded_label,
                                                  self.dset_info[dset_name],
                                                  show_hists=show_hists,
                                                  show_imgs=show_imgs,
                                                  res=resolution,
                                                  save=save)

                        if accumulate:
                            accumulator.append(proc_return)
                except Exception as e:
                    print(e)
                    #raise ValueError
            res_dict[resolution] = accumulator
        if accumulate:
            return proc_dir, res_dict