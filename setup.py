import os
import shutil
import tensorflow as tf

def download_test_data():
  url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')

  dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
  train_dir = os.path.join(dataset_dir, 'train')
  remove_dir = os.path.join(train_dir, 'unsup')
  shutil.rmtree(remove_dir)

def main():
  download_test_data()

if __name__ == "__main__":
  main()
