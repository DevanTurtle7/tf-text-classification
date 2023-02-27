import tensorflow as tf
from train import custom_standardization

def main():
  model = tf.keras.models.load_model('saved_model')
  examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
  ]

  print(model.predict(examples))

if __name__ == '__main__':
  main()