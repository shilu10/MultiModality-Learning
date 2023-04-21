import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

class DataPreprocessor: 
    def __init__(self): 
        pass 

    def dataframe_to_tensor_dataset(self, dataframe):
        columns = ["image_1_path", "image_2_path", "text_1", "text_2", "label_idx"]
        dataframe = dataframe[columns].copy()
        labels = dataframe.pop("label_idx")
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.shuffle(buffer_size=len(dataframe))
        return ds

    def reshape_image(image_path, img_shape):
        extension = tf.strings.split(image_path)[-1]

        image = tf.io.read_file(image_path)
        if extension == b"jpg":
            image = tf.image.decode_jpeg(image, 3)
        else:
            image = tf.image.decode_png(image, 3)
        image = tf.image.resize(image, img_shape)
        return image

    def get_encoder_input(roberta_preprocessor_model, text_1, text_2, roberta_input_features):
        text_1 = tf.convert_to_tensor([text_1])
        text_2 = tf.convert_to_tensor([text_2])
        encoder_inputs = roberta_preprocessor_model([text_1, text_2])
        encoder_inputs = {feature: tf.squeeze(encoder_inputs[feature]) for feature in roberta_input_features}
        return encoder_inputs

    def preprocess_text_and_image(sample, roberta_preprocessor_model, roberta_input_features, img_shape):
        image_1 = self.reshape_image(sample["image_1_path"], img_shape)
        image_2 = self.reshape_image(sample["image_2_path"], img_shape)
        encoder_inputs = self.get_encoder_input(roberta_preprocessor_model, sample["text_1"], sample["text_2"], roberta_input_features)
        return {"image_1": image_1, "image_2": image_2, "text": encoder_inputs}

    def prepare_dataset(dataframe, batch_size, auto, training=True):
        ds = self.dataframe_to_tensor_dataset(dataframe)
        if training:
            ds = ds.shuffle(len(train_df))
        ds = ds.map(lambda x, y: (self.preprocess_text_and_image(x, roberta_preprocessor_model, ROBERTA_INPUT_FEATURES, IMG_SHAPE), y)).cache()
        ds = ds.batch(batch_size).prefetch(auto)
        return ds