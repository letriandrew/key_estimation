import numpy as np
import librosa
import scipy.linalg
from typing import List
from scipy.stats import zscore
from dataclasses import dataclass 

@dataclass
class KeyEstimator:
    """
    Estimate key from pitch class distribution

    Parameters:
    X: np.ndarray, shape(12,)   
        Pitch-class distribution. Does not need to be normalized

    Returns:
    major : np.ndarray, shape(12,) 
    minor : np.ndarray, shape(12,) 

        for each key (C:maj, ..., B:maj) and (C:min, ... , B:min),
        the correlation score for 'X' against that key

    """

    # coefficients for Krumhansl and Schmuckle: https://rnhart.net/articles/key-finding/

    major = np.asarray(6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
    minor = np.asarray(6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

    #post constructor
    def __post_init__(self):
        self.major = zscore(self.major)
        self.major_norm = scipy.linalg.norm(self.major)
        self.major = scipy.linalg.circulant(self.major)

        self.minor = zscore(self.minor)
        self.minor_norm = scipy.linalg.norm(self.minor)
        self.minor = scipy.linalg.circulant(self.minor)

    def __call__(self, x: np.array) -> List[np.array]:
        x = zscore(x)
        x_norm = scipy.linalg.norm(x)

        coeffs_major = self.major.T.dot(x) / self.major_norm / x_norm 
        coeffs_minor = self.minor.T.dot(x) / self.minor_norm / x_norm

        return coeffs_major, coeffs_minor