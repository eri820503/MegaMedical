import imageio as io
from tqdm.notebook import tqdm_notebook
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class OCTA500:

    def __init__(self):
        self.name = "OCTA500"
        self.dset_info = {
            "OCTA_3M":{
                "main":"OCTA500",
                "image_root_dir":f"{paths['DATA']}/OCTA500/original_unzipped/retrieved_04_01/OCTA_3M/Projection Maps/OCT(FULL)",
                "label_root_dir":f"{paths['DATA']}/OCTA500/original_unzipped/retrieved_04_01/OCTA_3M/GroundTruth",
                "modality_names":["Retinal"],
                "planes":[0],
                "labels": [1,2,3],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            },
            "OCTA_6M":{
                "main":"OCTA500",
                "image_root_dir":f"{paths['DATA']}/OCTA500/original_unzipped/retrieved_04_01/OCTA_6M/Projection Maps/OCT(FULL)",
                "label_root_dir":f"{paths['DATA']}/OCTA500/original_unzipped/retrieved_04_01/OCTA_6M/GroundTruth",
                "modality_names":["Retinal"],
                "planes":[0],
                "labels": [1,2,3],
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
                  redo_processed=True):
        assert not(version is None and save), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = pps.make_processed_dir(self.name, dset_name, save, version, self.dset_info[dset_name])
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        accumulator = []
        for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            for image in image_list:
                try:
                    proc_dir_template = os.path.join(proc_dir, f"megamedical_v{version}", dset_name, "*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"
                        
                        if load_images:
                            loaded_image = io.imread(im_dir)
                            loaded_label = io.imread(label_dir)
                            assert not (loaded_label is None), "Invalid Label"
                            assert not (loaded_image is None), "Invalid Image"
                        else:
                            loaded_image = None
                            loaded_label = io.imread(label_dir)

                        proc_return = proc_func(proc_dir,
                                              version,
                                              dset_name,
                                              image, 
                                              loaded_image,
                                              loaded_label,
                                              self.dset_info[dset_name],
                                              show_hists=show_hists,
                                              show_imgs=show_imgs,
                                              save=save)

                    if accumulate:
                        accumulator.append(proc_return)
            except Exception as e:
                print(e)
                #raise ValueError
        if accumulate:
            return proc_dir, accumulator