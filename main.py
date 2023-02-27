import tensorflow as tf
from train import custom_standardization


def main():
  model = tf.keras.models.load_model('saved_model')
  print('\nEnter a comment to be analyzed. Press enter to quit.')
  print('\n> ', end='')
  response = input()

  while response != '':
    print(model.predict([response]))
    print('\n> ', end='')
    response = input()


if __name__ == '__main__':
  main()