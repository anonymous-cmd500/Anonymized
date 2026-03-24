"""
EEG Preprocessing for ZuCo Dataset (Pickle Format)

Loads the pre-built pickle files produced by the existing pipeline and
extracts word-level frequency-band EEG features.  Also provides per-channel
z-score normalisation computed on the training split.

The existing pickle format stores sentences as dicts:
{
    'content': str,                          # sentence text
    'word': [                                # list of word dicts
        {
            'content': str,                  # word string
            'word_level_EEG': {
                'GD': {
                    'GD_t1': np.ndarray(105,),
                    'GD_t2': ..., 'GD_a1': ..., 'GD_a2': ...,
                    'GD_b1': ..., 'GD_b2': ..., 'GD_g1': ..., 'GD_g2': ...
                }
            }
        }, ...
    ],
    'sentence_level_EEG': { ... }
}
"""

import os
import pickle
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


def load_pickle_datasets(
    data_dir: str,
    task_files: List[str],
) -> List[Dict]:
    """
    Load one or more ZuCo pickle task files.

    Args:
        data_dir:   Directory containing the pickle files.
        task_files: List of filenames (e.g. ["task1-SR-dataset.pickle", ...]).

    Returns:
        List of dataset dicts, each mapping subject → list[sentence_dict].
    """
    datasets = []
    for fname in task_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            print(f"[WARNING] Pickle file not found, skipping: {fpath}")
            continue
        with open(fpath, "rb") as f:
            datasets.append(pickle.load(f))
        print(f"[INFO] Loaded {fpath}")
    return datasets


class EEGPreprocessor:
    """
    Extracts word-level EEG feature vectors from ZuCo pickle sentence dicts
    and applies per-channel z-score normalisation.

    Feature vector per word:
        concat([band_1(105), band_2(105), ..., band_K(105)])
        → shape (105 * K,)     (default K=8 → dim 840)

    Normalisation:
        Computed on the *training* split only (call ``fit`` first),
        then applied via ``transform``.
    """

    def __init__(
        self,
        eeg_type: str = "GD",
        bands: Optional[List[str]] = None,
        n_channels: int = 105,
    ):
        self.eeg_type = eeg_type
        self.bands = bands or ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]
        self.n_channels = n_channels
        self.feature_dim = n_channels * len(self.bands)

        # Normalisation statistics (set by ``fit``)
        self._mean: Optional[np.ndarray] = None   # (feature_dim,)
        self._std: Optional[np.ndarray] = None     # (feature_dim,)

    # ── Feature extraction ──────────────────────────────────────────────
    def extract_word_features(self, word_obj: dict) -> Optional[np.ndarray]:
        """
        Extract concatenated frequency-band features for a single word.

        Returns:
            np.ndarray of shape (feature_dim,) or None if data is missing/bad.
        """
        try:
            parts = []
            for band in self.bands:
                key = f"{self.eeg_type}{band}"
                arr = word_obj["word_level_EEG"][self.eeg_type][key]
                parts.append(np.asarray(arr, dtype=np.float32))
            vec = np.concatenate(parts)
        except (KeyError, TypeError):
            return None

        if vec.shape[0] != self.feature_dim:
            return None
        if np.isnan(vec).all():
            return None
        # Replace remaining NaN with 0 (channel mean will be applied later)
        vec = np.nan_to_num(vec, nan=0.0)
        return vec

    def extract_sentence(self, sent_obj: dict) -> Optional[Tuple[np.ndarray, str]]:
        """
        Extract all word features for one sentence.

        Returns:
            (eeg_matrix, sentence_text) where eeg_matrix has shape (num_words, feature_dim)
            or None if the sentence is unusable.
        """
        if sent_obj is None:
            return None
        text = sent_obj.get("content", "")
        words = sent_obj.get("word", [])
        if not words or not text:
            return None

        vecs = []
        for w in words:
            v = self.extract_word_features(w)
            if v is None:
                return None  # drop entire sentence if any word is bad
            vecs.append(v)

        eeg = np.stack(vecs, axis=0)  # (num_words, feature_dim)

        # Reject sentences where >50 % of values are zero/NaN
        if (eeg == 0).sum() > 0.5 * eeg.size:
            return None

        return eeg, text

    # ── Normalisation ───────────────────────────────────────────────────
    def fit(self, eeg_matrices: List[np.ndarray]):
        """
        Compute per-feature mean and std from a list of (num_words, feature_dim)
        arrays (training split only).
        """
        all_vecs = np.concatenate(eeg_matrices, axis=0)  # (total_words, feature_dim)
        self._mean = all_vecs.mean(axis=0)
        self._std = all_vecs.std(axis=0)
        # Prevent division by zero for dead channels
        self._std[self._std < 1e-8] = 1.0
        print(f"[EEGPreprocessor] Fit normalisation on {all_vecs.shape[0]} word vectors")

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        """Z-score normalise using fitted statistics. Input shape (num_words, D)."""
        if self._mean is None:
            raise RuntimeError("Call fit() before transform()")
        return (eeg - self._mean) / self._std

    def fit_transform(self, eeg_matrices: List[np.ndarray]) -> List[np.ndarray]:
        self.fit(eeg_matrices)
        return [self.transform(m) for m in eeg_matrices]

    # ── Convenience: extract entire dataset from pickle dicts ───────────
    def extract_all_sentences(
        self,
        dataset_dicts: List[Dict],
        subject: str = "ALL",
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Walk through one or more task dataset dicts and return
        a list of (eeg_matrix, sentence_text) pairs.
        """
        samples: List[Tuple[np.ndarray, str]] = []
        for ds in dataset_dicts:
            subjects = list(ds.keys()) if subject == "ALL" else [subject]
            for subj in subjects:
                if subj not in ds:
                    continue
                for sent_obj in ds[subj]:
                    result = self.extract_sentence(sent_obj)
                    if result is not None:
                        samples.append(result)
        print(f"[EEGPreprocessor] Extracted {len(samples)} usable sentences")
        return samples

    def extract_all_sentences_with_subjects(
        self,
        dataset_dicts: List[Dict],
        subject: str = "ALL",
    ) -> List[Tuple[np.ndarray, str, str]]:
        """
        Like extract_all_sentences but returns (eeg_matrix, text, subject_id).
        Subject ID is needed for subject-level train/test splitting.
        """
        samples: List[Tuple[np.ndarray, str, str]] = []
        for ds in dataset_dicts:
            subjects = list(ds.keys()) if subject == "ALL" else [subject]
            for subj in subjects:
                if subj not in ds:
                    continue
                for sent_obj in ds[subj]:
                    result = self.extract_sentence(sent_obj)
                    if result is not None:
                        eeg, text = result
                        samples.append((eeg, text, subj))
        print(f"[EEGPreprocessor] Extracted {len(samples)} usable sentences with subject info")
        return samples

    # ── Serialisation ───────────────────────────────────────────────────
    def save_stats(self, path: str):
        np.savez(path, mean=self._mean, std=self._std)

    def load_stats(self, path: str):
        data = np.load(path)
        self._mean = data["mean"]
        self._std = data["std"]
