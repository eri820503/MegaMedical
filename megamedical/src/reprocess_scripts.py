def get_all_datasets(dataset_ver, data_root="/home/vib9/src/data"):
    proper_dsets = []
    for dset in os.listdir(data_root):
        dset_dir = os.path.join(data_root, dset, f"processed/megamedical_v{dataset_ver}")
        if os.path.isdir(dset_dir):
            for sub_dset in os.listdir(dset_dir):
                proper_dsets.append(f"{dset}~{sub_dset}")
    return proper_dsets


def make_splits(datasets, dataset_ver, root="/share/sablab/nfs02/users/gid-dalcaav/data/v3.2-midslice"):
    for dataset in datasets:
        print(f"STARTING SPLITTING {dataset}")
        info = dataset.split("~")
        sub_dset_p = os.path.join(info[0], info[1])
        sub_dset_full_path = os.path.join(root, sub_dset_p)
        for modality in os.listdir(sub_dset_full_path):
            mod_dir = os.path.join(sub_dset_full_path, modality)
            save_names = np.array(list(filter(lambda val: val != "train.txt" and val !=
                                      "val.txt" and val != "test.txt", os.listdir(mod_dir))))
            #save_names = np.array(
            #    list(map(lambda val: os.path.join(sub_dset_p, modality, val), clean_names)))
            
            total_amount = len(save_names)
            indices = np.arange(total_amount)
            np.random.shuffle(indices)

            train_amount = int(total_amount*0.6)
            val_test_amount = total_amount - train_amount
            val_amount = int(val_test_amount*0.5)
            test_amount = val_test_amount - val_amount

            train_indices = indices[:train_amount]
            val_indices = indices[train_amount:train_amount+val_amount]
            test_indices = indices[-test_amount:]

            train_names = save_names[train_indices]
            val_names = save_names[val_indices]
            test_names = save_names[test_indices]
            
            #print(train_names)
            #raise ValueError
            
            train_file = open(os.path.join(mod_dir, "train.txt"), "w")
            val_file = open(os.path.join(mod_dir, "val.txt"), "w")
            test_file = open(os.path.join(mod_dir, "test.txt"), "w")

            for file_name in train_names:
                train_file.write(file_name + "\n")
            train_file.close()

            for file_name in val_names:
                val_file.write(file_name + "\n")
            val_file.close()

            for file_name in test_names:
                test_file.write(file_name + "\n")
            test_file.close()

        print(f"DONE SPLITTING {dataset}!")


def repackage_npz_to_npy(datasets, dataset_ver, root="/home/vib9/src/data"):
    for dataset in datasets:
        print(f"STARTING SMALLIFYING {dataset}")
        info = dataset.split("~")
        sub_dset_root = os.path.join(
            root, info[0], f"processed/megamedical_v{dataset_ver}", info[1])
        for modality in os.listdir(sub_dset_root):
            modality_subdset_root = os.path.join(sub_dset_root, modality)
            for path in os.listdir(modality_subdset_root):
                path_file = os.path.join(modality_subdset_root, path)
                path = os.path.join(path_file, f'{info[1]}|{modality}|{path}|resize_data_256.npz') 
                if os.path.isfile(path):
                    data_256 = np.load(path)
                    image_256 = data_256["image"]
                    seg_256 = data_256["seg"]
                    linfo_256 = data_256["linfo"]

                    data_128 = np.load(str(path).replace("256","128"))
                    image_128 = data_128["image"]
                    seg_128 = data_128["seg"]
                    linfo_128 = data_128["linfo"]

                    np.save(str(path).replace("256.npz","256_image.npy"), image_256)
                    np.save(str(path).replace("256.npz","256_seg.npy"), seg_256)
                    np.save(str(path).replace("256.npz","256_linfo.npy"), linfo_256)
                    np.save(str(path).replace("256.npz","128_image.npy"), image_128)
                    np.save(str(path).replace("256.npz","128_seg.npy"), seg_128)
                    np.save(str(path).replace("256.npz","128_linfo.npy"), linfo_128)

                    os.remove(path)
                    os.remove(str(path).replace("256","128"))

        print(f"DONE SWAPPING NPZ -> NPY {dataset}!")
        

def produce_new_res(datasets, dataset_ver, root ="/share/sablab/nfs02/users/gid-dalcaav/data/v3.2-midslice"):
    for dataset in datasets:
        print(f"STARTING MIDIFYING {dataset}")
        info = dataset.split("~")
        sub_dset_root = os.path.join(root, info[0], info[1])
        for modality in os.listdir(sub_dset_root):
            modality_subdset_root = os.path.join(sub_dset_root, modality)
            for path in os.listdir(modality_subdset_root):
                if path not in ["train.txt", "val.txt", "test.txt"]:
                    path_file = os.path.join(modality_subdset_root, path)
                    
                    img_128_name = glob.glob(os.path.join(path_file, "resize_image_128_*"))[0]
                    seg_128_name = glob.glob(os.path.join(path_file, "resize_seg_128_*"))[0]

                    image_128_volume = np.load(img_128_name, mmap_mode='r')
                    seg_128_volume = np.load(seg_128_name, mmap_mode='r')
                    
                    """
                    f, axarr = plt.subplots(nrows=1, ncols=2)
                    if seg_128_volume.shape[2] != 1:
                        axarr[0].imshow(image_128_volume)
                        axarr[1].imshow(np.argmax(seg_128_volume, axis=2))
                    else:
                        axarr[0].imshow(image_128_volume)
                        axarr[1].imshow(seg_128_volume)
                        
                    plt.show()
                    """

                    image_sigma_64 = (0.5, 0.5)
                    seg_sigma_64 = (0.5, 0.5, 0)

                    blurred_image_128 = ndimage.gaussian_filter(image_128_volume, sigma=image_sigma_64)
                    blurred_seg_128 = ndimage.gaussian_filter(seg_128_volume, sigma=seg_sigma_64)

                    image_zoom_tup_64 = (0.5, 0.5)
                    seg_zoom_tup_64 = (0.5, 0.5, 1)

                    img_64_volume = ndimage.zoom(blurred_image_128, zoom=image_zoom_tup_64, order=1)
                    seg_64_volume = ndimage.zoom(blurred_seg_128, zoom=seg_zoom_tup_64, order=0)
                    
                    """
                    f, axarr = plt.subplots(nrows=1, ncols=2)
                    if seg_128_volume.shape[2] != 1:
                        axarr[0].imshow(img_64_volume)
                        axarr[1].imshow(np.argmax(seg_64_volume, axis=2))
                    else:
                        axarr[0].imshow(img_64_volume)
                        axarr[1].imshow(seg_64_volume)
                    
                    plt.show()
                    """
                    def repl_last(s, sub, repl):
                        index = s.rfind(sub)
                        if index == -1:
                            return s
                        return s[:index] + repl + s[index+len(sub):]

                    img_64_name = repl_last(img_128_name, "128", "64")
                    seg_64_name = repl_last(seg_128_name, "128", "64")

                    #print(img_64_name)
                    #print(seg_64_name)
                
                    np.save(img_64_name, img_64_volume)
                    np.save(seg_64_name, seg_64_volume)
                
        print(f"DONE MIDIFYING {dataset}!")
        

def produce_pickl_files(datasets, dataset_ver, root ="/home/vib9/src/data"):
    for dataset in datasets:
        try:
            print(f"STARTING TRANSFERING PIKL FOR {dataset}")
            info = dataset.split("~")
            sub_dset_root = os.path.join(root, info[0], f"processed/megamedical_v{dataset_ver}", info[1])
            midslice_dset_root = os.path.join(root, info[0], f"processed/midslice_v{dataset_ver}", info[1])

            for modality in os.listdir(midslice_dset_root):

                modality_subdset_root = os.path.join(sub_dset_root, modality)
                midslice_mod_root = os.path.join(midslice_dset_root, modality)

                for path in os.listdir(midslice_mod_root):
                    if not(path in ["train.txt", "val.txt", "test.txt"]):
                        path_file = os.path.join(modality_subdset_root, path)
                        new_midslice_file = os.path.join(midslice_mod_root, path)

                        pickle_path = os.path.join(new_midslice_file, "label_info_dict.pickle")
                        new_pickle_path = os.path.join(path_file, "label_info_dict.pickle")
                        if not os.path.exists(new_pickle_path):
                            shutil.copyfile(pickle_path, new_pickle_path)
        except:
            print(f"Error in {dataset}")
               
        print(f"DONE TRANSFERING PIKL FOR {dataset}!")
        

def check_processed(datasets, dataset_ver, root="/home/vib9/src/data"):
    invalid_subdsets = []
    missing_files = 0
    for dataset in datasets:
        info = dataset.split("~")
        sub_dset_root = os.path.join(root, info[0], f"processed/megamedical_v{dataset_ver}", info[1])
        for modality in os.listdir(sub_dset_root):
            modality_subdset_root = os.path.join(sub_dset_root, modality)
            clean_names = list(filter(lambda val: val != "train.txt" and val !=
                                      "val.txt" and val != "test.txt", os.listdir(modality_subdset_root)))
            for path in clean_names:
                path_file = os.path.join(modality_subdset_root, path)
                path = os.path.join(path_file, f'{info[1]}|{modality}|{path}|resize_data_256.npz')
                
                img_256_path = str(path).replace("256.npz","256_image.npy")
                seg_256_path = str(path).replace("256.npz","256_seg.npy")
                linfo_256_path = str(path).replace("256.npz","256_linfo.npy")
                
                img_128_path = str(path).replace("256.npz","128_image.npy")
                seg_128_path = str(path).replace("256.npz","128_seg.npy")
                linfo_128_path = str(path).replace("256.npz","128_linfo.npy") 
                
                files = [img_256_path, seg_256_path, linfo_256_path, img_128_path, seg_128_path, linfo_128_path]
                for file in files:
                    if not os.path.isfile(file):
                        missing_files += 1
                        print(f"Missing!: {file}")
                        if not dataset in invalid_subdsets:
                            invalid_subdsets.append(dataset)
                            break
                            
        print(f"PROBLEMATIC DATASETS: {invalid_subdsets}!, MISSING FILES: {missing_files}")


def process_datasets(dataset_ver, resize, generate_splits, repackage_files, check_dsets, mid_datasets, ppf, list_to_proc):
    assert dataset_ver is not None, "Must choose a version manually."
    if generate_splits:
        make_splits(list_to_proc, dataset_ver)
    if repackage_files:
        repackage_npz_to_npy(list_to_proc, dataset_ver)
    if check_dsets:
        check_processed(list_to_proc, dataset_ver)
    if mid_datasets:
        produce_midslice_dataset(list_to_proc, dataset_ver)
    if ppf:
        produce_pickl_files(list_to_proc, dataset_ver)
    if resize:
        produce_new_res(list_to_proc, dataset_ver)


if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        dataset = sys.argv[1]
        ltp = []
        for sub_dataset in os.listdir(os.path.join("/share/sablab/nfs02/users/gid-dalcaav/data/v3.2-midslice",dataset)):
            ltp.append(f"{dataset}~{sub_dataset}")
    else:
        #ltp = get_all_datasets(dataset_ver="3.1")
        # Test dataset
        ltp = [""]
    
    dataset_ver = None

    process_datasets(dataset_ver,
                     resize=False,
                     generate_splits=True,
                     repackage_files=False,
                     check_dsets=False,
                     mid_datasets=False,
                     ppf=False,
                     list_to_proc=ltp)