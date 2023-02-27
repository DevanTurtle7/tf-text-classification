import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')

  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def vectorize_text(vectorize_layer, text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


def train():
  batch_size = 32
  seed = 42

  raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
  )

  raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
  )

  raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size
  )

  max_features = 10000
  sequence_length = 250

  vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
  )

  train_text = raw_train_ds.map(lambda x, y: x)
  vectorize_layer.adapt(train_text)

  """
  text_batch, label_batch = next(iter(raw_train_ds))
  first_review, first_label = text_batch[0], label_batch[0]
  print('Review', first_review)
  print('Label', raw_train_ds.class_names[first_label])
  print('Vectorized review', vectorize_text(vectorize_layer, first_review, first_label))
  """

  vectorize_lambda = lambda text, label: vectorize_text(vectorize_layer, text, label)
  train_ds = raw_train_ds.map(vectorize_lambda)
  val_ds = raw_val_ds.map(vectorize_lambda)
  test_ds = raw_test_ds.map(vectorize_lambda)

  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

  embedding_dim = 16

  model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
  ])

  model.summary()

  model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True), 
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
  )

  epochs = 30
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  loss, accuracy = model.evaluate(test_ds)

  print("Loss: ", loss)
  print("Accuracy: ", accuracy)

  """
  history_dict = history.history
  history_dict.keys()

  acc = history_dict['binary_accuracy']
  val_acc = history_dict['val_binary_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')

  plt.show()
  """
  export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
  ])

  export_model.compile(
      loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
  )

  loss, accuracy = export_model.evaluate(raw_test_ds)
  print(accuracy)
  export_model.save('saved_model')


def main():
  train()


if __name__ == '__main__':
  main()