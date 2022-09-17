import imageio as io
from tqdm.notebook import tqdm_notebook
import glob
import gzip
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class STARE:

    def __init__(self):
        self.name = "STARE"
        self.dset_info = {
            "retrieved_2021_12_06":{
                "main": "STARE",
                "image_root_dir":f"{paths['DATA']}/STARE/original_unzipped/retrieved_2021_12_06/images",
                "label_root_dir":f"{paths['DATA']}/STARE/original_unzipped/retrieved_2021_12_06/labels",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
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
        proc_dir = os.path.join(paths['DATA'], self.name, "processed")
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        accumulator = []
        for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            try:
                proc_dir_template = os.path.join(proc_dir, f"midslice_v{version}", dset_name, "*", image)
                if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                    im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                    label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], f"{image[:-6]}vk.ppm.gz")
                    with gzip.open(im_dir,'r') as fin:        
                        loaded_image = io.imread(fin, as_gray=True)
                    with gzip.open(label_dir,'r') as fin2:        
                        loaded_label = io.imread(fin2, as_gray=True)

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
                                          resolutions=resolutions,
                                          save=save)

                    if accumulate:
                        accumulator.append(proc_return)
            except Exception as e:
                print(e)
                #raise ValueError
        if accumulate:
            return proc_dir, accumulator