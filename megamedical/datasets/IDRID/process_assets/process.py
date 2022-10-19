from PIL import Image
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class IDRID:

    def __init__(self):
        self.name = "IDRID"
        self.dset_info = {
            "retreived_2022_03_04":{
                "main": "IDRID",
                "image_root_dir":f"{paths['DATA']}/IDRID/original_unzipped/retreived_2022_03_04/A. Segmentation/1. Original Images/a. Training Set",
                "label_root_dir":f"{paths['DATA']}/IDRID/original_unzipped/retreived_2022_03_04/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            }
        }

    def proc_func(self,
                  subdset,
                  task,
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
        subj_dict, res_dict = proc.process_image_list(process_IDRID_image,
                                                      proc_dir,
                                                      task,
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
        
        
global process_IDRID_image
def process_IDRID_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        rtp = item["resolutions"] if item['redo_processed'] else put.check_proc_res(item)
        if len(rtp) > 0:
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            lab_root = dset_info[item['subdset']]["label_root_dir"]
            ma_dir = os.path.join(lab_root, "1. Microaneurysms", f"{item['image'][:-4]}_MA.tif")
            he_dir = os.path.join(lab_root, "2. Haemorrhages", f"{item['image'][:-4]}_HE.tif")
            ex_dir = os.path.join(lab_root, "3. Hard Exudates", f"{item['image'][:-4]}_EX.tif")
            se_dir = os.path.join(lab_root, "4. Soft Exudates", f"{item['image'][:-4]}_SE.tif")
            od_dir = os.path.join(lab_root, "5. Optic Disc", f"{item['image'][:-4]}_OD.tif")

            loaded_image = np.array(Image.open(im_dir).convert('L'))

            loaded_label = np.array(Image.open(ma_dir))
            if os.path.exists(he_dir):
                he = np.array(Image.open(he_dir))*2
                loaded_label += he
            if os.path.exists(ex_dir):
                ex = np.array(Image.open(ex_dir))*3
                loaded_label += ex
            if os.path.exists(se_dir):
                se = np.array(Image.open(se_dir))*4
                loaded_label += se
            if os.path.exists(od_dir):
                od = np.array(Image.open(od_dir))*5
                loaded_label += od

            assert not (loaded_image is None), "Invalid Image"
            assert not (loaded_label is None), "Invalid Label"

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