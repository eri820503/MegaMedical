import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import os

#New line!
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
                        if dset_name == "VerSe19":
                            im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{image}_ct.nii.gz")
                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_seg-vert_msk.nii.gz")
                        else:
                            im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{image}_dir-ax_ct.nii.gz")
                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_dir-ax_seg-vert_msk.nii.gz")

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