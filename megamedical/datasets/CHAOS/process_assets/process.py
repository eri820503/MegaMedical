from PIL import Image
import pydicom as dicom
import numpy as np
from tqdm.notebook import tqdm_notebook
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class CHAOS:

    def __init__(self):
        self.name = "CHAOS"
        self.dset_info = {
            "CT":{
                "main":"CHAOS",
                "image_root_dir":f"{paths['DATA']}/CHAOS/original_unzipped/retreived_2022_03_08/Train_Sets/CT",
                "label_root_dir":f"{paths['DATA']}/CHAOS/original_unzipped/retreived_2022_03_08/Train_Sets/CT",
                "modality_names":["CT"],
                "planes":[0],
                "labels": [1,2,3],
                "clip_args": [600,1500],
                "norm_scheme": "CT"
            },
            "MR":{
                "main":"CHAOS",
                "image_root_dir":f"{paths['DATA']}/CHAOS/original_unzipped/retreived_2022_03_08/Train_Sets/MR",
                "label_root_dir":f"{paths['DATA']}/CHAOS/original_unzipped/retreived_2022_03_08/Train_Sets/MR",
                "modality_names":["T2"],
                "planes":[0],
                "labels": [1,2,3],
                "clip_args": [0.5, 99.5],
                "norm_scheme": "MR"
            }
        }

    def proc_func(self,
                  subdset,
                  pps_function,
                  parallelize=False,
                  load_images=True,
                  accumulate=False,
                  version=None,
                  show_imgs=False,
                  save=False,
                  show_hists=False,
                  resolutions=None,
                  redo_processed=True):
        assert not(version is None and save), "Must specify version for saving."
        assert subdset in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = os.path.join(paths['ROOT'], "processed")
        image_list = sorted(os.listdir(self.dset_info[subdset]["image_root_dir"]))
        subj_dict, res_dict = proc.process_image_list(process_CHAOS_image,
                                                      proc_dir,
                                                      image_list,
                                                      parallelize,
                                                      pps_function,
                                                      resolutions,
                                                      self.name,
                                                      subdset,
                                                      self.dset_info,
                                                      redo_processed,
                                                      load_images,
                                                      show_hists,
                                                      version,
                                                      show_imgs,
                                                      accumulate,
                                                      save)
        if accumulate:
            return proc_dir, subj_dict, res_dict
        
        
global process_CHAOS_image
def process_CHAOS_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed']:
            rtp = put.check_proc_res(item)
        else:
            rtp = item["resolutions"]
        if len(rtp) > 0:
            if item['subdset'] == "CT":
                DicomDir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'], "DICOM_anon")
                GroundDir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'], "Ground")
            else:
                DicomDir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'], "T2SPIR/DICOM_anon")
                GroundDir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'], "T2SPIR/Ground")

            if item['load_images']:
                planes = []
                for plane in os.listdir(DicomDir):
                    planes.append(dicom.dcmread(os.path.join(DicomDir, plane)).pixel_array)
                loaded_image = np.stack(planes)

                gt_planes = []
                for gt_plane in os.listdir(GroundDir):
                    gt_planes.append(np.array(Image.open(os.path.join(GroundDir, gt_plane)).convert('L')))
                loaded_label = np.stack(gt_planes)

                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                gt_planes = []
                for gt_plane in os.listdir(GroundDir):
                    gt_planes.append(np.array(Image.open(os.path.join(GroundDir, gt_plane)).convert('L')))
                loaded_label = np.stack(gt_planes)

            # Set the name to be saved
            subj_name = item['image'].split(".")[0]
            pps_function = item['pps_function']
            proc_return = pps_function(item['proc_dir'],
                                        item['version'],
                                        item['subdset'],
                                        subj_name, 
                                        loaded_image,
                                        loaded_label,
                                        dset_info[item['subdset']],
                                        show_hists=item['show_hists'],
                                        show_imgs=item['show_imgs'],
                                        resolutions=rtp,
                                        save=item['save'])

            return proc_return, subj_name
        else:
            return None, None
    except Exception as e:
        print(e)
        return None, None