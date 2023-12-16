import logging
from typing import List

from torch import int32, long, zeros

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    audio = zeros(
        len(dataset_items),
        get_max_length(dataset_items, 'audio')
    )

    audio_class = zeros(len(dataset_items), dtype=long)
    audio_path = []

    for idx, item in enumerate(dataset_items):
        item_audio = item['audio']
        item_audio_class = 0 if item['is_spoofed'] else 1
        item_audio_path = item['audio_path']

        item_audio_length = item_audio.shape[-1]

        audio[idx, :item_audio_length] = item_audio
        audio_class[idx] = item_audio_class
        audio_path.append(item_audio_path)

    return {
        'audio': audio,
        'audio_class': audio_class,
        'audio_path': audio_path,
    }

def get_max_length(dataset_items, element):
    return max([item[element].shape[-1] for item in dataset_items])