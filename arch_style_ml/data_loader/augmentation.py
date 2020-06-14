import numpy as np
import numbers
import abc
from PIL.Image import Image
from typing import Callable

import torchvision.transforms as T
import torchvision.transforms.functional as F

ImageTransformer = Callable[[Image], Image]


def get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class NewPad(object):
    def __init__(self, fill=0, padding_mode="constant"):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(padding={0}, fill={1}, padding_mode={2})".format(
                self.fill, self.padding_mode
            )
        )


DEFAULT_TRAIN_TRANSFORM: ImageTransformer = T.Compose(
    [
        NewPad(299),
        # T.RandomResizedCrop(299),
        T.Resize(299),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

DEFAULT_VAL_TRANSFORM: ImageTransformer = T.Compose(
    [
        NewPad(299),
        T.Resize(299),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class AugmentationFactoryBase(abc.ABC):
    def __init__(
        self, train_transform: ImageTransformer, test_transform: ImageTransformer
    ):
        self.train_transform = train_transform
        self.test_transform = test_transform


class ArchStyleTransforms(AugmentationFactoryBase):
    def __init__(
        self,
        train_transform: ImageTransformer = DEFAULT_TRAIN_TRANSFORM,
        test_transform: ImageTransformer = DEFAULT_VAL_TRANSFORM,
    ):
        self.train_transform = train_transform
        self.test_transform = test_transform
