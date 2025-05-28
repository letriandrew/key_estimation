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

    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

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
    
def main(): 
    # read open the midi file and extract the pitch class distribution
    # then, estimate the key of the piece using the Krumhansl-Schumcuckler key estimation algorithim

    # load the audio as a waveform 'y'
    # store the sampling rate as 'sr'

    file = input("Enter path to audio file: ")
    y, sr = librosa.load(file)

    # compute chromagram (energy in each pitch class at each frame of the audio)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

    # sum the energy across all the frames
    # this gives us energy in each pitch class over the entire song
    pitch_class_distribution = chromagram.sum(axis=1)

    # estimate the key using K and S
    key_estimator = KeyEstimator()
    major, minor = key_estimator(pitch_class_distribution)

    mapping = {0: "C", 1: "C#/Db", 2: "D", 3: "D#/Eb", 4: "E", 5: "F", 6: "F#/Gb", 7: "G", 8: "G#/Ab", 9: "A", 10: "A#/Bb", 11: "B"}

    # print the results 
    print("\nMajor key coefficients:")
    for i, coeff in enumerate(major):
        print(f"{mapping[i]}:\t{coeff:.2f}")

    print("\nMinor key coefficients:")
    for i, coeff in enumerate(minor):
        print(f"{mapping[i]}:\t{coeff:.2f}")

    # print estimated key
    print(f"Estimated key: {mapping[np.argmax(major)]} major {mapping[np.argmax(minor)]} minor")
    print("Most likely:", mapping[np.argmax(major)] + "major" if np.max(major) > np.max(minor) else mapping[np.argmax(minor)] + "minor")

if __name__ == "__main__":
    main()