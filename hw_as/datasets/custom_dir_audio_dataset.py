import logging
import os
from os import path
from pathlib import Path

import torchaudio

from hw_as.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, *args, **kwargs):
        index = []
        index = [
            # mixed paths
            # target paths
            # ref paths
            # clear id
            # noise id
            # mix length
            
        ]
        index = []
        audio_dir = Path(audio_dir)
        spoofed_dir = audio_dir / 'spoofed'
        og_dir = audio_dir / 'original'

        parts = [spoofed_dir, og_dir]
        for part in parts:
            for root, dirs, files in os.walk(str(part), topdown=False, followlinks=True):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    entry = {
                        "path": file_path,
                        "is_spoofed": str(part).endswith('spoofed'),
                    }
                    index.append(entry)

        super().__init__(index, *args, **kwargs)
