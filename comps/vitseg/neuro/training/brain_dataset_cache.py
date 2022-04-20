import collections
import sys
import numpy as np
import torch

from typing import Any, Callable, List, Union, Sequence, Optional
from torch.utils.data import Subset
from monai.data import CacheDataset

from generator_coords import CoordsGenerator


class BrainCacheDataset(CacheDataset):
    """General purpose dataset class with several data sources `list_data`."""

    def __init__(
            self,
            data: Sequence,
            transform: Union[Sequence[Callable], Callable],
            rand_transform: Union[Sequence[Callable], Callable],
            list_shape: List[int],
            list_sub_shape: List[int],
            num_workers: Optional[int] = None,
            cache_rate: float = 1.0,
            cache_num: int = sys.maxsize,
            n_subvolumes: int = None,
            copy_cache: bool = True,
            progress: bool = True,
            mode: str = "train",
            input_key: str = "images",
            output_key: str = "nii_labels",
    ):
        """
        Args:
            list_data (List[Dict]): list of dicts, that stores
                you data annotations,
                (for example path to images, labels, bboxes, etc.)
            transform (callable): transforms to use on dict.
                (for example normalize image, add blur, crop/resize/etc)
            list_shape (List[int]):
            list_sub_shape (List[int]):
        """
        super().__init__(data, transform, cache_num, cache_rate,
                         num_workers, progress, copy_cache)
        self.generator = CoordsGenerator(
            list_shape=list_shape, list_sub_shape=list_sub_shape
        )
        self.mode = mode
        self.subvolume_shape = np.array(list_sub_shape)
        self.subjects = len(self.data)
        self.n_subvolumes = n_subvolumes
        self.input_key = input_key
        self.output_key = output_key
        self._rand_transform = rand_transform

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return self.n_subvolumes * len(self.data)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Any:
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            List of elements by index
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        batch_list = []
        subject_id = index // self.n_subvolumes

        subj_data_dict = self._transform(subject_id)
        if self.mode == 'train' and self._rand_transform:
            subj_data_dict = self._rand_transform(subj_data_dict)

        coords_index = index % self.n_subvolumes
        coords = self.generator.get_coordinates(mode="test")

        if self.mode in ["train", "validation"] or coords_index >= len(coords):
            coords = self.generator.get_coordinates()
        else:
            coords = np.expand_dims(coords[coords_index], 0)

        batch_list.append(
            self.__crop__(subj_data_dict, coords))

        return batch_list

    def __crop__(self, dict_, coords):
        """Get crop of images.
        Args:
            dict_ (List[Dict]): list of dicts, that stores
                you data annotations,
                (for example path to images, labels, bboxes, etc.)
            coords (callable): coords of crops

        Returns:
            crop images
        """
        output = {}
        output_labels_list = []
        output_images_list = []
        for start_end in coords:
            for key, dict_key in dict_.items():
                if key == self.input_key:
                    output_images_list.append(
                            dict_key[
                                :,
                                start_end[0][0] : start_end[0][1],
                                start_end[1][0] : start_end[1][1],
                                start_end[2][0] : start_end[2][1],
                            ],
                    )

                elif key == self.output_key:
                    output_labels_list.append(
                            dict_key[
                                :,
                                start_end[0][0] : start_end[0][1],
                                start_end[1][0] : start_end[1][1],
                                start_end[2][0] : start_end[2][1],
                            ],
                    )

        output_images = torch.cat(output_images_list)
        output_labels = torch.cat(output_labels_list)
        output[self.input_key] = output_images
        output[self.output_key] = output_labels.squeeze()
        output["coords"] = coords
        return output
