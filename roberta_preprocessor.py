from tensorflow import keras 
import tensorflow as tf 
import tensorflow_hub as hub 


def build_roberta_preprocessor(preprocessor_path):
    # Reference : https://tfhub.dev/jeongukjae/roberta_en_cased_L-12_H-768_A-12/1
    """Returns Model mapping string features to BERT inputs.

      Args:
        sentence_features: A list with the names of string-valued features.
        seq_length: An integer that defines the sequence length of BERT inputs.

      Returns:
        A Keras Model that can be called on a list or dict of string Tensors
        (with the order or names, resp., given by sentence_features) and
        returns a dict of tensors for input to BERT.
  """
    
    text_inputs = [
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tf.keras.layers.Input(shape=(), dtype=tf.string),
    ]
    
    #roberta preprocessor
    preprocessor = hub.load(preprocessor_path)
    tokenize = hub.KerasLayer(preprocessor.tokenize)
    bert_pack_inputs = hub.KerasLayer(preprocessor.bert_pack_inputs)
    
    tokenized_inputs = [tokenize(segment) for segment in text_inputs]
    encoder_inputs = bert_pack_inputs(tokenized_inputs)
    
    return keras.Model(text_inputs, encoder_inputs)

