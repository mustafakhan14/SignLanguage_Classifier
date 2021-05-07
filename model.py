import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

train_path = Path('../Data/asl_alphabet_train')
test_path = Path('../Data/asl_alphabet_test')

train_filepaths = list(train_path.glob(r'**/*.jpg'))
test_filepaths = list(test_path.glob(r'**/*.jpg'))

"""
Read in images from path and store in data-frame with each image
concatenated to its label. Returns a shuffled data-frame.
"""
def extract_data(pathname):
    labels = [str(pathname[i]).split("/")[-2] for i in range(len(pathname))]
    path = pd.Series(pathname, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    df = pd.concat([path, labels], axis=1)
    return df.sample(frac=1).reset_index(drop=True)

train_set = extract_data(train_filepaths)
test_set = extract_data(test_filepaths)