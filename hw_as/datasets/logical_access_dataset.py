import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_as.base.base_dataset import BaseDataset
from hw_as.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y", 
}

PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

class LogicalAccessDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "logical_access"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "LA.zip"
        print(f"Loading Logical Access dataset")
        download_file(URL_LINKS["dataset"], arch_path)
        print('Unpacking files')
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LA").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        shutil.rmtree(str(self._data_dir / "LA"))


    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / f'ASVspoof2019_LA_{part}'
        if not split_dir.exists():
            self._load_dataset()

        print('Making index')
        protocol_path = self._data_dir / 'ASVspoof2019_LA_cm_protocols' / PROTOCOLS[part]
        index = []
        with open(str(protocol_path), 'r') as fp:
            n_lines = sum([1 for _ in fp])

        with open(str(protocol_path), 'r') as fp:
            for line in tqdm(fp, total=n_lines):
                line = line.strip()
                SpeakerID, UtteranceID, UtteranceType, SpoofAlgoId, IsSpoofed = line.split()
                audio_path = split_dir / 'flac' / f'{UtteranceID}.flac'
                audio_info = torchaudio.info(str(audio_path))
                length = audio_info.num_frames / audio_info.sample_rate
                index.append(
                    {
                        "path": str(audio_path.absolute().resolve()),
                        "speaker_id": SpeakerID,
                        "is_spoofed": IsSpoofed != 'bonafide',
                        "audio_len": length,
                    }
                )

        return index