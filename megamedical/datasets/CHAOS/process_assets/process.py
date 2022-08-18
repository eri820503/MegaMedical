from PIL import Image
import pydicom as dicom
from tqdm import tqdm
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class CHAOS:

    def __init__(self):
        self.name = "CHAOS"
        self.dset_info = {
            "CT":{
                "main":"CHAOS",
                "image_root_dir":f"{paths['DATA']}/CHAOS/processed/original_unzipped/retreived_2022_03_08/Train_Sets/CT",
                "label_root_dir":f"{paths['DATA']}/CHAOS/processed/original_unzipped/retreived_2022_03_08/Train_Sets/CT",
                "modality_names":["CT"],
                "planes":[0],
                "clip_args": [600,1500],
                "norm_scheme": "CT",
                "do_clip": True,
                "proc_size":256
            },
            "MR":{
                "main":"CHAOS",
                "image_root_dir":f"{paths['ROOT']}/megamedical/datasets/CHAOS/processed/original_unzipped/retreived_2022_03_08/Train_Sets/MR",
                "label_root_dir":f"{paths['ROOT']}/megamedical/datasets/CHAOS/processed/original_unzipped/retreived_2022_03_08/Train_Sets/MR",
                "modality_names":["T2"],
                "planes":[0],
                "clip_args": None,
                "norm_scheme": "MR",
                "do_clip": True,
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
                        if dset_name == "CT":
                            DicomDir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "DICOM_anon")
                            GroundDir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "Ground")
                        else:
                            DicomDir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "T2SPIR/DICOM_anon")
                            GroundDir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "T2SPIR/Ground")

                        planes = []
                        for plane in os.listdir(DicomDir):
                            planes.append(dicom.dcmread(os.path.join(DicomDir, plane)).pixel_array)
                        loaded_image = np.stack(planes)

                        gt_planes = []
                        for gt_plane in os.listdir(GroundDir):
                            gt_planes.append(np.array(Image.open(os.path.join(GroundDir, gt_plane)).convert('L')))
                        loaded_label = np.stack(gt_planes)

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