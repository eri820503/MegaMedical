import nibabel as nib
from tqdm.notebook import tqdm_notebook
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


class COCA:

    def __init__(self):
        self.name = "COCA"
        self.dset_info = {
            "-":{
                "main":"COCA",
                "image_root_dir": None,
                "label_root_dir": None,
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args":None,
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
        image_list = os.listdir(self.dset_info[dset_name]["label_root_dir"])
        proc_dir = os.path.join(paths['ROOT'], "processed")
        accumulator = []
        res_dict = {}
        for resolution in resolutions:
            for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
                image = image.split(".")[0]
                try:
                    # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
                    template_root = os.path.join(proc_dir, f"res{resolution}", self.name)
                    mid_proc_dir_template = os.path.join(template_root, f"midslice_v{version}", dset_name, "*/*", image)
                    max_proc_dir_template = os.path.join(template_root, f"maxslice_v{version}", dset_name, "*/*", image)
                    if redo_processed or (len(glob.glob(mid_proc_dir_template)) == 0) or (len(glob.glob(max_proc_dir_template)) == 0):
                        #flat_img_slices = glob.glob(os.path.join(self.dset_info[dset_name]["image_root_dir"],image,"*/*"))
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],f"{image}.xml")
                        keypoints = process_xml(label_dir)

                        #loaded_image = np.stack([dicom.dcmread(image_path) for image_path in flat_img_slices])

                        loaded_label = None
                        #loaded_label = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, ".xml")

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
