import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class FeTA:

    def __init__(self):
        self.name = "FeTA"
        self.dset_info = {
            "retrieved_2022_02_16":{
                "main":"FeTA",
                "image_root_dir":f"{paths['DATA']}/FeTA/original_unzipped/retrieved_2022_02_16/affine_subjs",
                "label_root_dir":f"{paths['DATA']}/FeTA/original_unzipped/retrieved_2022_02_16/affine_subjs",
                "modality_names":["MRI"],
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
                    if redo_processed or (len(glob.glob(mid_proc_dir_template)) == 0) or (len(glob.glob(max_proc_dir_template)) == 0):
                        im_id = image[4:7]

                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{im_id}_img.nii.gz")
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{im_id}_seg.nii.gz")

                        if load_images:
                            loaded_image = put.resample_nib(nib.load(im_dir))
                            loaded_label = put.resample_mask_to(nib.load(label_dir), loaded_image)

                            loaded_image = loaded_image.get_fdata()
                            loaded_label = loaded_label.get_fdata()
                            assert not (loaded_label is None), "Invalid Label"
                            assert not (loaded_image is None), "Invalid Image"
                        else:
                            loaded_image = None
                            loaded_label = nib.load(label_dir).get_fdata()

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