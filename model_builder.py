import tensorflow.keras as keras 
import tensorflow as tf 
import tensorflow_hub as hub 
import pandas as pd 
import numpy as np 


class ModelBuilder:
    
    def __init__(self):
        pass 

    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = keras.layers.Dense(projection_dims)(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            x = keras.layers.Add()([projected_embeddings, x])
            projected_embeddings = keras.layers.LayerNormalization()(x)
        return projected_embeddings


    def create_vision_encoder(self, num_projection_layers, projection_dims, dropout_rate, trainable=False):

        resnet_v2 = keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", pooling="avg"
        )
        
        for layer in resnet_v2.layers:
            layer.trainable = trainable

        # Receive the images as inputs.
        image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
        image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

        preprocessed_1 = keras.applications.resnet_v2.preprocess_input(image_1)
        preprocessed_2 = keras.applications.resnet_v2.preprocess_input(image_2)

        embeddings_1 = resnet_v2(preprocessed_1)
        embeddings_2 = resnet_v2(preprocessed_2)
        embeddings = keras.layers.Concatenate()([embeddings_1, embeddings_2])

        outputs = self.project_embeddings(
            embeddings, num_projection_layers, projection_dims, dropout_rate
        )
        return keras.Model([image_1, image_2], outputs, name="vision_encoder")


    def create_text_encoder(self, roberta_encoder_path, bert_input_features, 
                                    num_projection_layers, projection_dims, dropout_rate, trainable=False):
        roberta = hub.KerasLayer(roberta_encoder_path, name="bert",)
        roberta.trainable = trainable
        
        text_inputs = {
            feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
            for feature in bert_input_features
        }

        embeddings = roberta(text_inputs)["pooled_output"]

        outputs = self.project_embeddings(
            embeddings, num_projection_layers, projection_dims, dropout_rate
        )
        return keras.Model(text_inputs, outputs, name="text_encoder")


    def create_multimodal_model(self, roberta_encoder_path, img_input_dims, bert_input_features, num_projection_layers=1, 
                                        projection_dims=256, dropout_rate=0.1, vision_trainable=False, text_trainable=False, rate=0.2):
        
        image_1 = keras.Input(shape=img_input_dims, name="image_1")
        image_2 = keras.Input(shape=img_input_dims, name="image_2")

        text_inputs = {
            feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
            for feature in bert_input_features
        }

        vision_encoder = self.create_vision_encoder(
            num_projection_layers, projection_dims, dropout_rate, vision_trainable
        )
        text_encoder = self.create_text_encoder(
            roberta_encoder_path, bert_input_features, num_projection_layers, projection_dims, dropout_rate, text_trainable
        )
        
        vision_projections = vision_encoder([image_1, image_2])
        text_projections = text_encoder(text_inputs)
        vision_projections = keras.layers.Dropout(rate)(vision_projections)
        text_projections = keras.layers.Dropout(rate)(text_projections)
        
        query_value_attention_seq = keras.layers.Attention(use_scale=True, dropout=0.2)([vision_projections, text_projections])

        concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
        contextual = keras.layers.Concatenate()([concatenated, query_value_attention_seq])

        outputs = keras.layers.Dense(3, activation="softmax")(contextual)
        return keras.Model([image_1, image_2, text_inputs], outputs)


