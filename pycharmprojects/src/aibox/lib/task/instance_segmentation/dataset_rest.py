import io
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple, List, Optional, Dict, Any

import lmdb
import numpy as np
import torch.utils.data.dataset
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

from .preprocessor import Preprocessor
from ...augmenter import Augmenter
from ...bbox import BBox


class Dataset(torch.utils.data.dataset.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        VAL = 'val'
        TEST = 'test'
        UNION = 'union'

    @dataclass
    class Annotation:
        @dataclass
        class Object:
            name: str
            difficulty: bool
            bbox: BBox
            mask_color: int

        filename: str
        image_id: str
        image_width: int
        image_height: int
        image_depth: int
        objects: List[Object]

    @dataclass
    class Item:
        path_to_image: str
        image_id: str
        image: Tensor
        processed_image: Tensor
        bboxes: Tensor
        processed_bboxes: Tensor
        masks: Tensor
        processed_masks: Tensor
        classes: Tensor
        difficulties: Tensor
        process_dict: Dict[str, Any]

    ItemTuple = Tuple[str, str, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]









    @staticmethod


class ConcatDataset(torch.utils.data.dataset.ConcatDataset):


        if self.mode == self.Mode.TRAIN:
            image_ids = read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'train.txt'))
        elif self.mode == self.Mode.VAL:
            image_ids = read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'val.txt'))
        elif self.mode == self.Mode.TEST:
            image_ids = read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'test.txt'))
        elif self.mode == self.Mode.UNION:
            image_ids = []
            image_ids += read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'train.txt'))
            image_ids += read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'val.txt'))
            image_ids += read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'test.txt'))
            image_ids = list(sorted(set(image_ids)))
        else:
            raise ValueError('Invalid mode')

        self.annotations = []

        for image_id in image_ids:
            path_to_annotation_xml = os.path.join(self._path_to_annotations_dir, f'{image_id}.xml')
            tree = ET.ElementTree(file=path_to_annotation_xml)
            root = tree.getroot()

            annotation = self.Annotation(
                filename=root.find('filename').text,
                image_id=image_id,
                image_width=int(root.find('size/width').text),
                image_height=int(root.find('size/height').text),
                image_depth=int(root.find('size/depth').text),
                objects=[]
            )

            for tag_object in root.iterfind('object'):
                mask_color_tag = tag_object.find('mask/color')
                if mask_color_tag is not None:
                    annotation.objects.append(
                        self.Annotation.Object(
                            name=tag_object.find('name').text,
                            difficulty=tag_object.find('difficult').text == '1',
                            bbox=BBox(
                                left=float(tag_object.find('bbox/left').text),
                                top=float(tag_object.find('bbox/top').text),
                                right=float(tag_object.find('bbox/right').text),
                                bottom=float(tag_object.find('bbox/bottom').text)
                            ),
                            mask_color=int(mask_color_tag.text)
                        )
                    )

            if exclude_difficulty:
                annotation.objects = [it for it in annotation.objects if not it.difficulty]

            if len(annotation.objects) > 0:  # skip annotations without any objects
                self.annotations.append(annotation)

        with open(path_to_meta_json, 'r') as f:
            self.category_to_class_dict = json.load(f)
            self.class_to_category_dict = {v: k for k, v in self.category_to_class_dict.items()}

        self._lmdb_env: Optional[lmdb.Environment] = None
        self._lmdb_txn: Optional[lmdb.Transaction] = None

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Union[Item, ItemTuple]:
        annotation = self.annotations[index]
        image_id = annotation.image_id
        path_to_image = os.path.join(self._path_to_images_dir, annotation.filename)

        mask_colors = sorted([obj.mask_color for obj in annotation.objects if obj.mask_color != 0])
        assert mask_colors == list(set(mask_colors))

        path_to_mask_image = os.path.join(self._path_to_segmentations_dir, f'{annotation.image_id}.png')
        mask_image = Image.open(path_to_mask_image)
        mask_image = np.array(mask_image, dtype=np.uint8)
        mask_image = torch.from_numpy(mask_image)

        bboxes = self._mask_image_to_bboxes(mask_image, mask_colors)  # get bbox by mask image instead of which in annotation
        masks = self._mask_image_to_masks(mask_image, mask_colors)

        classes = [self.category_to_class_dict[obj.name] for obj in annotation.objects if obj.name != 'background']
        difficulties = [obj.difficulty for obj in annotation.objects if obj.name != 'background']

        if self._lmdb_txn is not None:
            binary = self._lmdb_txn.get(annotation.filename.encode())
            with io.BytesIO(binary) as f, Image.open(f) as image:
                image = to_tensor(image)
        else:
            with Image.open(path_to_image).convert('RGB') as image:
                image = to_tensor(image)

        processed_image, process_dict = self.preprocessor.process(image,
                                                                  is_train_or_eval=self.mode == self.Mode.TRAIN)

        processed_bboxes = bboxes.clone()
        processed_bboxes[:, [0, 2]] *= process_dict[Preprocessor.PROCESS_KEY_WIDTH_SCALE]
        processed_bboxes[:, [1, 3]] *= process_dict[Preprocessor.PROCESS_KEY_HEIGHT_SCALE]

        processed_mask_image = F.interpolate(
            input=mask_image.unsqueeze(dim=0).unsqueeze(dim=0).float(),
            scale_factor=(process_dict[Preprocessor.PROCESS_KEY_HEIGHT_SCALE], process_dict[Preprocessor.PROCESS_KEY_WIDTH_SCALE]),
            mode='nearest',  # use mode `nearest` to avoid interpolated value
            recompute_scale_factor=True
        ).squeeze(dim=0).squeeze(dim=0).type_as(mask_image)
        processed_mask_image = F.pad(input=processed_mask_image,
                                     pad=[0, process_dict[Preprocessor.PROCESS_KEY_RIGHT_PAD], 0, process_dict[Preprocessor.PROCESS_KEY_BOTTOM_PAD]])  # pad has format [left, right, top, bottom]

        if self.augmenter is not None:
            processed_image, processed_bboxes, processed_mask_image, classes, difficulties, mask_colors = \
                self.augmenter.apply(processed_image, processed_bboxes, processed_mask_image,
                                     classes=classes, difficulties=difficulties, mask_colors=mask_colors)

        processed_masks = self._mask_image_to_masks(processed_mask_image, mask_colors)

        assert len(processed_bboxes) == len(processed_masks) == len(classes) == len(difficulties)

        if processed_masks.shape[1] != processed_image.shape[1] or processed_masks.shape[2] != processed_image.shape[2]:
            if abs(processed_masks.shape[1] - processed_image.shape[1]) <= 1 and abs(processed_masks.shape[2] - processed_image.shape[2]) <= 1:
                processed_masks = F.interpolate(
                    input=processed_masks.unsqueeze(dim=0).float(),
                    size=(processed_image.shape[1], processed_image.shape[2]),
                    mode='nearest'  # use mode `nearest` to avoid interpolated value
                ).squeeze(dim=0).type_as(processed_masks)
            else:
                raise ValueError

        classes = torch.tensor(classes, dtype=torch.long)
        difficulties = torch.tensor(difficulties, dtype=torch.int8)

        if not self.returns_item_tuple:
            return Dataset.Item(path_to_image, image_id, image, processed_image, bboxes, processed_bboxes, masks, processed_masks, classes, difficulties, process_dict)
        else:
            return path_to_image, image_id, image, processed_image, bboxes, processed_bboxes, masks, processed_masks, classes, difficulties, process_dict

    def setup_lmdb(self) -> bool:
        path_to_lmdb_dir = os.path.join(self.path_to_data_dir, 'lmdb')
        if os.path.exists(path_to_lmdb_dir):
            self._lmdb_env = lmdb.open(path_to_lmdb_dir)
            self._lmdb_txn = self._lmdb_env.begin()
            return True
        else:
            return False

    def teardown_lmdb(self):
        if self._lmdb_env is not None:
            self._lmdb_env.close()

    def _mask_image_to_bboxes(self, mask_image: Tensor, mask_colors: List[int]) -> Tensor:
        bboxes = []
        for mask_color in mask_colors:
            pos = (mask_image == mask_color).nonzero()
            if pos.shape[0] > 0:
                left = pos[:, 1].min().item()
                top = pos[:, 0].min().item()
                right = pos[:, 1].max().item()
                bottom = pos[:, 0].max().item()
                bboxes.append([left, top, right, bottom])
        bboxes = torch.tensor(bboxes, dtype=torch.float)
        return bboxes

    def _mask_image_to_masks(self, mask_image: Tensor, mask_colors: List[int]) -> Tensor:
        mask_colors = torch.tensor(mask_colors, dtype=torch.uint8)
        masks = mask_image.repeat(mask_colors.shape[0], 1, 1)
        masks = (masks == mask_colors.view(-1, 1, 1).expand_as(masks)).type_as(masks)
        return masks

    def num_classes(self) -> int:
        return len(self.class_to_category_dict)

    @staticmethod
    def collate_fn(item_tuple_batch: List[ItemTuple]) -> Tuple[ItemTuple]:
        return tuple(item_tuple_batch)


class ConcatDataset(torch.utils.data.dataset.ConcatDataset):

    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)
        assert len(datasets) > 0

        dataset: Dataset = self.datasets[0]

        for i in range(1, len(datasets)):
            assert dataset.class_to_category_dict == datasets[i].class_to_category_dict
            assert dataset.category_to_class_dict == datasets[i].category_to_class_dict
            assert dataset.num_classes() == datasets[i].num_classes()

        self.master = dataset