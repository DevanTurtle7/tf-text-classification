import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

def download_test_data():
  url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
  dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
  train_dir = os.path.join(dataset_dir, 'train')
  remove_dir = os.path.join(train_dir, 'unsup')
  shutil.rmtree(remove_dir)

def setup_test_data():
  batch_size = 32
  seed = 42

  raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
  )

def main():
  setup_test_data()

if __name__ == '__main__':
  main()