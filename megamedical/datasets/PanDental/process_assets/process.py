from PIL import Image
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class PanDental:

    def __init__(self):
        self.name = "PanDental"
        self.dset_info = {
            "v1":{
                "main": "PanDental",
                "image_root_dir":f"{paths['DATA']}/PanDental/original_unzipped/v1/Images",
                "label_root_dir":f"{paths['DATA']}/PanDental/original_unzipped/v1/orig_masks",
                "modality_names":["XRay"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None
            },
            "v2":{
                "main": "PanDental",
                "image_root_dir":f"{paths['DATA']}/PanDental/original_unzipped/v2/Images",
                "label_root_dir":f"{paths['DATA']}/PanDental/original_unzipped/v2/Segmentation1",
                "modality_names":["XRay"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None
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
        skip_list = ["106.png", "107.png", "109.png", "111.png", "112.png",
                         "113.png", "116.png", "19.png", "39.png", "64.png", "65.png",
                         "68.png", "70.png", "76.png", "78.png", "79.png", "98.png"]
        if dset_name == "v2":
            for item in skip_list:
                image_list.remove(item)
        image_list = sorted(image_list)
        
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
                    if redo_processed or (len(glob.glob(mid_proc_dir_template)) == 0) or (len(glob.glob(max_proc_dir_template)) == 0):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image)

                        loaded_image = np.array(Image.open(im_dir).convert('L'))

                        if dset_name == "v1":
                            loaded_label = np.array(Image.open(label_dir))[...,0]
                        else:
                            loaded_label = (np.array(Image.open(label_dir).convert('L')) > 0).astype(int)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

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