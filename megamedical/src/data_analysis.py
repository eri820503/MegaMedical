import seaborn as sns
import os
from tqdm.notebook import tqdm_notebook
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
    

            

        
        

