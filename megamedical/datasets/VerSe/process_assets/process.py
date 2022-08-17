import nibabel as nib
from tqdm import tqdm
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
                "image_root_dir":f"{paths['DATA']}/VerSe/processed/original_unzipped/VerSe19/dataset-verse19training/rawdata",
                "label_root_dir":f"{paths['DATA']}/VerSe/processed/original_unzipped/VerSe19/dataset-verse19training/derivatives",
                "modality_names":["CT"],
                "planes": [0],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            },
            "VerSe20":{
                "main": "VerSe",
                "image_root_dir":f"{paths['DATA']}/VerSe/processed/original_unzipped/VerSe20/dataset-01training/rawdata",
                "label_root_dir":f"{paths['DATA']}/VerSe/processed/original_unzipped/VerSe20/dataset-01training/derivatives",
                "modality_names":["CT"],
                "planes": [0, 1],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
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
                    proc_dir_template = os.path.join(proc_dir, f"megamedical_v{version}", dset_name, "*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):

                        if dset_name == "VerSe19":
                            im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{image}_ct.nii.gz")
                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_seg-vert_msk.nii.gz")

                            loaded_image = np.array(nib.load(im_dir).dataobj)
                            loaded_label = np.array(nib.load(label_dir).dataobj)
                        else:
                            im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{image}_dir-ax_ct.nii.gz")
                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_dir-ax_seg-vert_msk.nii.gz")

                            loaded_image = put.resample_nib(nib.load(im_dir))
                            loaded_label = np.array(put.resample_mask_to(nib.load(label_dir), loaded_image).dataobj)
                            loaded_image = np.array(loaded_image.dataobj)

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