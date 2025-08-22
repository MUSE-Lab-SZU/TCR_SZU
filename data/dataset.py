import os
import gc
import json
import bisect
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from atom.utils.logger import logger

class CachedTensorDataset(Dataset):
    """
    A PyTorch Dataset that loads data from multiple .pt files on demand,
    with an LRU cache at the sample level to limit memory usage.

    Each .pt file is expected to contain a dict with:
      - 'data': a list of tensors
      - 'num': the number of examples in this file (optional)
    """
    def __init__(
        self, 
        data_folder, 
        start=0,
        end=None,
        cache_size=100000,
        try_load_all = False, 
        sequential_read=True,
    ):
        """
        Args:
            data_folder (str): Path to folder containing .pt files.
            cache_size (int): Maximum number of samples to keep in cache.
            sequential_read (bool): If True, samples are read in sequential order.
        """
        self.data_folder = data_folder
        self.cache_size = cache_size
        self.sequential_read = sequential_read
        # Cache: map global idx -> tensor
        self.cache = OrderedDict()

        # Gather all .pt files and compute cumulative sizes
        pt_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.pt')])
        if end is None: end = len(pt_files)

        build_meta_info = False
        if os.path.exists(os.path.join(data_folder, "meta_info.json")):
            meta_info = json.load(open(os.path.join(data_folder, "meta_info.json"), "r"))
            meta_info = sorted(meta_info, key=lambda x: x["file"])
            if [info["file"] for info in meta_info] != pt_files:
                logger.info("meta info中的文件信息与实际文件夹中不同，重新构建")
                build_meta_info = True
        else:
            logger.info("未找到meta info，重新构建")
            build_meta_info = True

        if build_meta_info:
            meta_info = []
            for file in pt_files:
                logger.info(f"开始处理{file}")
                content = torch.load(os.path.join(data_folder, file))
                content.pop("data")
                meta_info.append(content)
            json.dump(meta_info, open(os.path.join(data_folder, "meta_info.json"), "w"))

        self.meta_info = meta_info[start:end]
        self.file_paths = [os.path.join(data_folder, f) for f in pt_files][start:end]
        self.file_counts = []
        cumulative = 0
        self.cum_counts = []
        for file in self.meta_info:
            num = file["frame_num"]
            self.file_counts.append(num)
            cumulative += num
            self.cum_counts.append(cumulative)
        self.total = cumulative
        logger.info(f"数据集总大小:{self.total}, 缓存大小:{self.cache_size}")
        if try_load_all:
            load_amount = min(self.cache_size, self.total)
            logger.info("尝试尽可能多的填充缓存")
            for index in tqdm(range(load_amount)):
                self.__getitem__(index)

    def __len__(self):
        return self.total

    def _get_file_index(self, idx):
        """
        Find which file contains the overall index `idx`.
        Returns file index file_idx.
        """
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(f"Index {idx} out of range (0, {self.total - 1})")
        file_idx = bisect.bisect_right(self.cum_counts, idx)
        return file_idx

    def __getitem__(self, idx):
        # If sample cached, return immediately
        if idx in self.cache:
            if not self.sequential_read:
                self.cache.move_to_end(idx)
            return self.cache[idx]

        # Determine which file holds this idx
        file_idx = self._get_file_index(idx)
        path = self.file_paths[file_idx]
        data_dict = torch.load(path, map_location='cpu')

        # Compute start index for this file
        start_idx = self.cum_counts[file_idx - 1] if file_idx > 0 else 0
        data_list = data_dict['data']

        # Cache all samples from this file
        for i, tensor in enumerate(data_list):
            global_idx = start_idx + i
            # Avoid overwriting if already cached
            if global_idx not in self.cache:
                self.cache[global_idx] = tensor
        
        # Evict oldest samples if cache exceeds limit
        if len(self.cache) > self.cache_size:
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            gc.collect()

        # Return the requested sample
        if not self.sequential_read:
            self.cache.move_to_end(idx)
        return self.cache[idx]

if __name__ == '__main__':
    dataset = CachedTensorDataset('/mnt/bn/vector2/tensors_320x180', cache_size=10000)
    print(dataset[1])