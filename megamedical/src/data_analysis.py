import seaborn as sns
import os
from tqdm.notebook import tqdm_notebook
import universandbox as usd
import numpy as np

from megamedical.utils.registry import paths
import megamedical.utils as utils
from megamedical.utils.proc_utils import *

def vis_label_hists(datasets, slice_version):
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])

    sns.set(rc = {'figure.figsize':(30,5)})
    for dataset in datasets:
        try:
            fig_path = os.path.join(paths["DATA"], dataset, "processed/figures")
            if slice_version:
                subdsets = os.listdir(fig_path)
                for subdset in subdsets:
                    subdset_fig_dir = os.path.join(fig_path, subdset)
                    planes = list(set([pf.split("_")[-1] for pf in os.listdir(subdset_fig_dir)]))
                    for plane in planes:
                        max_dict_path = os.path.join(subdset_fig_dir, f"max_lab_dict_{plane}")
                        middle_dict_path = os.path.join(subdset_fig_dir, f"mid_lab_dict_{plane}")
                        
                        total_maxslice_label_dict = load_obj(max_dict_path)
                        total_midslice_label_dict = load_obj(middle_dict_path)
                        
                        plane_key = int(plane[0])
                        ax1 = sns.barplot(x=list(total_midslice_label_dict[plane_key].keys()), y=list(total_midslice_label_dict[plane_key].values()))
                        ax1.set(title=f"Midslice | Plane {plane_key} | Label Frequency for {dataset}/{subdset}.")
                        ax1.bar_label(ax1.containers[0])
                        plt.show()
                        ax2 = sns.barplot(x=list(total_maxslice_label_dict[plane_key].keys()), y=list(total_maxslice_label_dict[plane_key].values()))
                        ax2.set(title=f"Maxslice | Plane {plane_key} | Label Frequency for {dataset}/{subdset}.")
                        ax2.bar_label(ax2.containers[0])
                        plt.show()
            else:
                dict_path = os.path.join(fig_path, "lab_dict.pickle")
                frequency_label_dict = load_obj(dict_path)
                sns.set(rc = {'figure.figsize':(30,5)})
                ax = sns.barplot(x=list(frequency_label_dict.keys()), y=list(frequency_label_dict.values()))
                ax.bar_label(ax.containers[0])
                ax.set(title=f"Label Frequency for {dataset}.")
                plt.show()
        except Exception as e:
            print(e)
            continue

            
def vis_dataset(datasets,
                num_vis,
                size=3,
                version="4.0"):
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])
        
    dataset_objects = [utils.build_dataset(ds) for ds in datasets]
    
    for d_idx, dataset in enumerate(datasets):
        try:
            maxslice_path = os.path.join(paths["DATA"], dataset, f"processed/maxslice_v{version}")
            midslice_path = os.path.join(paths["DATA"], dataset, f"processed/midslice_v{version}")
            subdsets = os.listdir(midslice_path)
            for subdset in subdsets:
                maxslice_modalities_dir = os.path.join(maxslice_path, subdset)
                midslice_modalities_dir = os.path.join(midslice_path, subdset)
                for modality in os.listdir(midslice_modalities_dir):
                    maxslice_subj_dir = os.path.join(maxslice_modalities_dir, modality)
                    midslice_subj_dir = os.path.join(midslice_modalities_dir, modality)
                    
                    chosen_subs = np.random.choice(os.listdir(midslice_subj_dir), num_vis)
                    
                    for plane in dataset_objects[d_idx].dset_info[subdset]["planes"]:
                    
                        midslice_imgs = [np.load(os.path.join(midslice_subj_dir,subj,f"img_128_{plane}.npy")) for subj in chosen_subs]
                        midslice_seg_volumes = [np.load(os.path.join(midslice_subj_dir,subj,f"seg_128_{plane}.npy")) for subj in chosen_subs]
                        if midslice_seg_volumes[0].shape[2] != 1:
                            midslice_segs = []
                            for msv in midslice_seg_volumes:
                                background = np.zeros((midslice_imgs[0].shape[0], midslice_imgs[0].shape[1], 1))
                                midseg_combined = np.argmax(np.concatenate([background, msv], axis=-1), axis=-1)
                                midslice_segs.append(midseg_combined)
                        else:
                            midslice_segs = [msv.squeeze() for msv in midslice_seg_volumes]
                        midslices = np.concatenate([midslice_imgs, midslice_segs])
                        
                        print(f"Mid-slice Dataset: {dataset}, Subdset: {subdset}, Modality: {modality}, Plane: {plane}")
                        usd.utils.display_array(midslices, 
                                                nrows=2, 
                                                ncols=num_vis,
                                                box_size=size,  
                                                do_colorbars=True, 
                                                cmap="gray")
                        plt.show()
                        
                        maxslice_img_volumes= [np.load(os.path.join(maxslice_subj_dir,subj,f"img_128_{plane}.npy")) for subj in chosen_subs]
                        maxslice_seg_volumes = [np.load(os.path.join(maxslice_subj_dir,subj,f"seg_128_{plane}.npy")) for subj in chosen_subs]
                        
                        for lab_idx in range(maxslice_img_volumes[0].shape[-1]):
                            print(f"Max-slice Dataset: {dataset}, Subdset: {subdset}, Modality: {modality}, Plane: {plane}, Label: {lab_idx}")
                            maxslice_imgs = np.array([miv[..., lab_idx] for miv in maxslice_img_volumes])
                            maxslice_segs = np.array([msv[..., lab_idx] for msv in maxslice_seg_volumes])

                            usd.utils.display_array(np.concatenate([maxslice_imgs, maxslice_segs]),  
                                                    nrows=2, 
                                                    ncols=num_vis,
                                                    box_size=size,  
                                                    do_colorbars=True, 
                                                    cmap="gray")
                            plt.show()
        except Exception as e:
            print(e)
            #raise ValueError
            continue
                    

            

        
        

