from omegaconf import DictConfig
from torchvision import transforms

from data.data_utils.transform import RandomSpatialAugmentor, EventPadderTransform
from .genx.build_dataset import build_prophesee_dataset

def make_transform(dataset_config: DictConfig, mode: str = 'train'):
    aug_config = dataset_config.augmentation
    augment = RandomSpatialAugmentor(
        h_flip_prob=aug_config.prob_hflip,
        rotation_prob=aug_config.rotate.prob,
        rotation_angle_range=(aug_config.rotate.min_angle_deg, aug_config.rotate.max_angle_deg),
        zoom_prob=aug_config.zoom.prob,
        zoom_in_weight=aug_config.zoom.zoom_in.weight,
        zoom_out_weight=aug_config.zoom.zoom_out.weight,
        zoom_in_range=(aug_config.zoom.zoom_in.factor.min, aug_config.zoom.zoom_in.factor.max),
        zoom_out_range=(aug_config.zoom.zoom_out.factor.min, aug_config.zoom.zoom_out.factor.max)
    )

    height, width = dataset_config.target_size
    padding = EventPadderTransform(target_height=height, target_width=width)

    if mode == 'train':
        transform = [augment, padding]
    elif mode == 'val':
        transform = [padding]
    elif mode == 'test':
        transform = [padding]

    return transforms.Compose([t for t in transform if t is not None])

def build_dataset(dataset_config: DictConfig, mode: str = 'train'):
    name = dataset_config.name
    transform = make_transform(dataset_config=dataset_config, mode=mode)

    if 'gen' in name:
        dataset = build_prophesee_dataset(dataset_config=dataset_config, mode=mode, transform=transform)

    else:
        print(f'{name=} not available')
        raise NotImplementedError
    
    return dataset