import nibabel as nib
from tqdm import tqdm
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
                "image_root_dir":f"{paths['DATA']}/FeTA/processed/original_unzipped/retrieved_2022_02_16/affine_subjs",
                "label_root_dir":f"{paths['DATA']}/FeTA/processed/original_unzipped/retrieved_2022_02_16/affine_subjs",
                "modality_names":["MRI"],
                "planes":[0, 1, 2],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            }
        }

    def proc_func(self,
                dset_name,
                  version=None,
                show_hists=False,
                  show_imgs=False,
                  save_slices=False,
                redo_processed=True):
        assert not(version is None and save_slices), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = pps.make_processed_dir(self.name, dset_name, save_slices, version, self.dset_info[dset_name])
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    proc_dir_template = os.path.join(proc_dir, f"megamedical_v{version}", dset_name, "*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                        im_id = image[4:7]

                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{im_id}_img.nii.gz")
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{im_id}_seg.nii.gz")

                        loaded_image = np.array(nib.load(im_dir).dataobj)
                        loaded_label = np.array(nib.load(label_dir).dataobj)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        pps.produce_slices(proc_dir,
                                          version,
                                          dset_name,
                                          image, 
                                          loaded_image,
                                          loaded_label,
                                          self.dset_info[dset_name],
                                          show_hists=show_hists,
                                          show_imgs=show_imgs,
                                          save_slices=save_slices)
                except Exception as e:
                    print(e)
                    #raise ValueError
                pbar.update(1)
        pbar.close()