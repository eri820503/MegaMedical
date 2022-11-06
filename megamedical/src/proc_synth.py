import megamedical.src.proc_funcs as pf

if __name__=="__main__":
    #Available steps: steps=["labels","stats","images","splits"]
    pf.process_pipeline(steps=["labels", "stats", "images"],
                    datasets=["Synthetic1000"],
                    ignore_datasets=None,
                    ignore_subdsets=None,
                    subdsets=None,
                    save=True,
                    slurm=False,
                    redo_processed=True,
                    resolutions=[64, 128, 256],
                    train_split=0.7,
                    version="4.0",
                    timeout=720,
                    mem_gb=64,
                    parallelize=False)
