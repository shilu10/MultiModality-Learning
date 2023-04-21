import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

class DataPreprocessor: 
    def __init__(self, roberta_preprocessor_model, roberta_input_features): 
        """
            this class provides the methods, that needed for the preprocessing of the Entailment dataset.
            Methods:
                dataframe_to_tensor_dataset
                reshape_image
                get_encoder_input
                preprocess_text_and_image
                prepare_dataset
            
            Attrs:
                roberta_preprocessor_model(type: keras.models.Model): roberta preprocessor model, which is need by
                                                 get_encoder_input method
                roberta_input_features(type: List): list of names of the input features of the roberta model.

        """
        self.roberta_preprocessor_model = roberta_preprocessor_model
        self.roberta_input_features = roberta_input_features

    def dataframe_to_tensor_dataset(self, dataframe):
        """
            this method will convert the pandas dataframe into the tensorflow dataset, for the better processing of the tpu or 
            gpu
            Params:
                dataframe(type: pandas.DataFrame): pandas dataframe, that needed to be converted.

            Return(type: tf.data.Dataset):
                tensorflow dataset, for the better performance.
        """
        columns = ["image_1_path", "image_2_path", "text_1", "text_2", "label_idx"]
        dataframe = dataframe[columns].copy()
        labels = dataframe.pop("label_idx")
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.shuffle(buffer_size=len(dataframe))
        return ds

    def reshape_image(self, image_path, img_shape):
        """
            this method, will read the image data from the image path, and reshape the image into the specified image
            shape.
            Params:
                image_path(type: str): Image Directory path for the tweet images.
                img_shape(type: tuple): Image shape, that the image needed to be resized.

            Return(type; np.ndarray):
                Numpy N-dimensional array.(image)

        """
        extension = tf.strings.split(image_path)[-1]

        image = tf.io.read_file(image_path)
        if extension == b"jpg":
            image = tf.image.decode_jpeg(image, 3)
        else:
            image = tf.image.decode_png(image, 3)
        image = tf.image.resize(image, img_shape)
        return image

    def get_encoder_input(self.text_1, text_2):
                                                            
        """
            this method, will input the data into roberta preprocessor model(bert model), and gets the embedding of the
            sentence.
            Params:
                text_1(type: str): text_1, the textual data that needed to converted as a embedding.
                text_2(type: str): text_2, the textual data that needed to converted as a embedding.
            
            Return(type: List)
                It returns the roberta preprocessed models output, which is the input of the roberta encoder model.
        """
        text_1 = tf.convert_to_tensor([text_1])
        text_2 = tf.convert_to_tensor([text_2])
        encoder_inputs = self.roberta_preprocessor_model([text_1, text_2])
        encoder_inputs = {feature: tf.squeeze(encoder_inputs[feature]) for feature in self.roberta_input_features}
        return encoder_inputs

    def preprocess_text_and_image(self, sample, img_shape):
        """
            this method will read the image convert the input image into specific image shape with the help of reshape_image 
            method. And also , it will get the embedding of the text1 and text2 in the dataframe.
            Params:
                sample(type: tf.data.Dataset): sample is the col of the tf dataset.
                img_shape(type: tuple): Image, input shape.
            
            Return(type: dict)
                this method, returns the dict of preprocessed image_1 and image_2 and encoder_input(out of the roberta preprocessor)
        """
        image_1 = self.reshape_image(sample["image_1_path"], img_shape)
        image_2 = self.reshape_image(sample["image_2_path"], img_shape)
        encoder_inputs = self.get_encoder_input(self.roberta_preprocessor_model, sample["text_1"], 
                                                            sample["text_2"], self.roberta_input_features)
        return {"image_1": image_1, "image_2": image_2, "text": encoder_inputs}

    def prepare_dataset(self, dataframe, batch_size, auto, training=True):
        """
            this method, will make sure to edit the tensorflow data datasets data, before passing it into the model. Which will
            convert the image_path into actual image data and textual data into embeddings.
            Params:
                roberta_preprocessor_model(type: keras.models.Model): roberta preprocessor model, which is need by
                                                 get_encoder_input method
                roberta_input_features(type: List): list of names of the input features of the roberta model.
                dataframe(type: pandas.DataFrame): DataFrame, that needed to preprocssed and converted into the dataset(tensorflow)
                batch_size(type: int): Batch Size.
                auto: prefetch auto type.
                training(type: bool): Specifying whether it is training or testing data.

            Return(type: tf.data.Dataset)
                this method returns the tensorflow dataset object after doing all the preprocessing works.
        """
        ds = self.dataframe_to_tensor_dataset(dataframe)
        if training:
            ds = ds.shuffle(len(train_df))
        ds = ds.map(lambda x, y: (self.preprocess_text_and_image(x, IMG_SHAPE), y)).cache()
        ds = ds.batch(batch_size).prefetch(auto)
        return ds