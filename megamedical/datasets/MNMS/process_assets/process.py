import nibabel as nib
from tqdm import tqdm
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class MNMS:

    def __init__(self):
        self.name = "MNMS"
        self.dset_info = {
            "2020":{
                "main":"MNMS",
                "image_root_dir":f"{paths['DATA']}/MNMS/processed/original_unzipped/2020/OpenDataset",
                "label_root_dir":f"{paths['DATA']}/MNMS/processed/original_unzipped/2020/OpenDataset",
                "modality_names":["T1"],
                "planes":[2],
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
        proc_dir = pps.make_processed_dir(self.name, dset_name, save_slices, version)
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{image}_sa.nii.gz")
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_sa_gt.nii.gz")

                        loaded_image = np.array(nib.load(im_dir).dataobj)[...,0]
                        loaded_label = np.array(nib.load(label_dir).dataobj)[...,0]

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
                                          show_imgs=show_imgs)
                except Exception as e:
                    print(e)
                pbar.update(1)
        pbar.close()