import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class MRSpectrogramDatasetLoader(Dataset):
    def __init__(self,
                 noisy_dir: str,
                 clean_dir: str,
                 segment_frames: int,
                 hop_frames: int = None,
                 pad_to_full_chunk: bool = False):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.segment_frames = segment_frames
        self.hop_frames = hop_frames if hop_frames is not None else segment_frames

        self.pad_to_full_chunk = pad_to_full_chunk

        self.file_list = sorted(os.listdir(clean_dir))
        self.cache: dict[str: dict[str, np.ndarray]] = {}
        self.index_map = []

        self._fill_index_map()

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        audio_filename, offset = self.index_map[idx]

        clean_path = os.path.join(self.clean_dir, audio_filename)
        noisy_path = os.path.join(self.noisy_dir, audio_filename)

        clean_spec = np.load(clean_path, allow_pickle=True)
        noisy_spec = np.load(noisy_path, allow_pickle=True)

        clean_audio_chunk = clean_spec[:, offset:offset + self.segment_frames, :]
        noisy_audio_chunk = noisy_spec[:, offset:offset + self.segment_frames, :]

        if clean_audio_chunk.shape[1] < self.segment_frames:
            pad_width = self.segment_frames - clean_audio_chunk.shape[1]
            clean_audio_chunk = np.pad(clean_audio_chunk, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            noisy_audio_chunk = np.pad(noisy_audio_chunk, ((0, 0), (0, pad_width), (0, 0)), mode='constant')

        return noisy_audio_chunk, clean_audio_chunk

    def _fill_index_map(self):
        print('Filling index map')

        skipped_files = []
        for npy_filename in tqdm(self.file_list):
            clean_path = os.path.join(self.clean_dir, npy_filename)
            noisy_path = os.path.join(self.noisy_dir, npy_filename)

            if os.path.isdir(clean_path) or not clean_path.endswith('.npy'):
                continue
            if os.path.isdir(noisy_path) or not noisy_path.endswith('.npy'):
                continue

            clean_spec = np.load(clean_path, allow_pickle=True)
            noisy_spec = np.load(noisy_path, allow_pickle=True)
            if clean_spec.shape != noisy_spec.shape:
                skipped_files.append(npy_filename)
                continue

            total_frames = clean_spec.shape[1]
            num_chunks = max(1, (total_frames - self.segment_frames) // self.hop_frames + 1)

            for i in range(num_chunks):
                offset = i * self.hop_frames
                self.index_map.append((npy_filename, offset))

            if self.pad_to_full_chunk and (total_frames - num_chunks * self.hop_frames) < self.segment_frames:
                final_offset = total_frames - self.segment_frames
                if final_offset > 0:
                    self.index_map.append((npy_filename, final_offset))

        if len(skipped_files) > 0:
            print(f'Skipped files: {skipped_files}')
