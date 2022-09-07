import seaborn as sns
import os
from tqdm.notebook import tqdm_notebook
import numpy as np

from megamedical.utils.registry import paths
import megamedical.utils as utils
from megamedical.utils.proc_utils import *

def vis_label_hists(datasets):
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])
        datasets.remove("RibSeg")
        datasets.remove("EchoNet")
        datasets.remove("SegThy")
        datasets.remove("TotalSeg")
    for dataset in datasets:
        try:
            dict_path = os.path.join(paths["DATA"], dataset, "processed/figures/lab_dict.pickle")
            frequency_label_dict = load_obj(dict_path)
            sns.set(rc = {'figure.figsize':(30,5)})
            ax = sns.barplot(x=list(frequency_label_dict.keys()), y=list(frequency_label_dict.values()))
            ax.bar_label(ax.containers[0])
            ax.set(title=f"Label Frequency for {dataset}.")
            plt.show()
        except Exception as e:
            print(e)
            continue
    

            

        
        

