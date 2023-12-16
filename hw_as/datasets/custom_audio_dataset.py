import logging
from pathlib import Path

import torchaudio

from hw_as.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            assert "path" in entry
            assert "is_spoofed" in entry
            for attr in entry.keys():
                if not 'path' in attr:
                    continue
                assert Path(entry[attr]).exists(), f"Path {entry[attr]} doesn't exist"
                entry[attr] = str(Path(entry[attr]).absolute().resolve())
            t_info = torchaudio.info(entry["path"])
            entry["audio_len"] = t_info.num_frames / t_info.sample_rate

        super().__init__(index, *args, **kwargs)
