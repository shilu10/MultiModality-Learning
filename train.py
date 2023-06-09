from focal_loss import SparseCategoricalFocalLoss
import tensorflow_addons as tfa
import tensorflow as tf 
import tensorflow.keras as keras 
from roberta_preprocessor import *
from model_builder import *
from sklearn.preprocessing import train_test_split
import argparse 
from data_preprocessor import *
from utils import *
import warnings, os
import pickle 


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--image-shape', description="Image Shape, to which images needed to be resized", default=(128, 128))      
parser.add_argument('--input-dims', description="Input dims of the image to model.", default=(128, 128, 3))      
parser.add_argument('--batch-size', description="Batch Size for the training data.", default=32)      
parser.add_argument('--save-path', description="Path to save the trained multimodal model.", default="models/")      
parser.add_argument('--save-model', description="Save the trained model", default=True)  
parser.add_argument('--image-dir', description="Directory path of the tweets image data", default="tweets_images/")      
parser.add_argument('--epoch', description="Number of epochs to train the model", default=10)      
parser.add_argument('--lr', description="Learning Rate for the optimizer", default=0.001)      
parser.add_argument('--opt-type', description="Optimizer for building the model, options are (sgd, adam)", default="adam")      
args = parser.parse_args()

if args.image_shape != args.(input_dims[0],args.input_dims[1]):
    raise ValueError("Image Shape and Input Dims shape should be same")

if not os.path.exists(args.image_dir): 
    raise FileNotFoundError("Image Directory or file is not exists")

if args.batch_size > 2000:
    raise ValueError("Batch Size should be lesser than <2000")

if args.opt_type not in ["adam", "sgd"]:
    raise ValueError("Specified optimizer is not supported")


ROBERTA_PREPROCESSOR_PATH = "https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1"
ROBERTA_ENCODER_PATH ="https://tfhub.dev/jeongukjae/roberta_en_cased_L-12_H-768_A-12/1"
LABEL_MAP = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}
IMAGE_DIR_PATH = 'tweet_images/'
IMG_SHAPE = args.image_shape
ROBERTA_INPUT_FEATURES = ["input_word_ids", "input_type_ids", "input_mask"]
BATCH_SIZE  = args.batch_size 
AUTO =  tf.data.AUTOTUNE
INPUT_DIMS = args.input_dims


def train_model(multimodal_model, train_labels, train_ds, val_ds): 
    
    """
        Main Function to train the MultiModality Model.
        Params:
            multimodal_model(type: keras.models.Model): Build and uncompiled keras model.
            train_labels(type: np.array): Classes in the dataset.
            train_ds(type: tf.data.Dataset): Training dataset of the model.
            val_ds(type: tf.data.Dataset): vaidation dataset of the model.

    """
    try: 
        train_labels_ohe = keras.utils.to_categorical(train_labels)
        class_totals = train_labels_ohe.sum(axis=0)
        class_weight = dict()

        for i in range(0, len(class_totals)):
            class_weight[i] = class_totals.max() / class_totals[i]
        
        opt = get_optimizer(lr, optimizer)
        
        multimodal_model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", 
                                                metrics="accuracy")
        multimodal_model.fit(train_ds, epochs=epochs, 
                                        validation_data=val_ds, batch_size=batch_size, 
                                        class_weight=class_weight)

        if args.save_model and args.save_path: 
            multimodal_model.save(save_path)

    except Exception as error: 
        return error

if __name__ == "-_main_ _": 

    model_builder = ModelBuilder()
    roberta_preprocessor_model = build_roberta_preprocessor(ROBERTA_PREPROCESSOR_PATH)
    data_preprocessor = DataPreprocessor(roberta_preprocessor_model, ROBERTA_INPUT_FEATURES)

    df = pd.read_csv(dataframe_file+"/tweets.csv")
    create_images_path(df)

    train_df, val_df = train_test_split(
        df, test_size=0.1, stratify=df["label"].values, random_state=42
    )

    train_df, test_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"].values, random_state=42
    )

    with open('test_df.obj', "wb") as f: 
        pickle.dump(test_df, f)

    train_ds = data_preprocessor.prepare_dataset(train_df, BATCH_SIZE, AUTO)
    test_ds = data_preprocessor.prepare_dataset(test_df, BATCH_SIZE, AUTO, False)
    val_ds = data_preprocessor.prepare_dataset(val_df, BATCH_SIZE, AUTO, False)

    multimodal_model = model_builder.create_multimodal_model(ROBERTA_ENCODER_PATH, INPUT_DIMS, ROBERTA_INPUT_FEATURES)                                            
    train_labels = get_labels(train_ds)
    test_labels = get_labels(test_ds)
    val_labels = get_labels(val_ds)

    train_model(multimodal_model, train_labels, train_ds, val_ds)