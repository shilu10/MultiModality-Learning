import os 

def get_data(dataframe_url, image_url):
    """
        this function, will download the entailment dataset and unzip it.
        Params:
            dataframe_url(type: str): https url path of the dataset.
            iamge_url(type: sttr): https url path of the image.
    """
    try:
        os.system(f"wget {dataframe_url} --quiet")
        os.system(f"wget {image_url} --quiet")
        os.system("tar -xzf tweet_images.tar.gz")

    except Exception as error:
        return error


def create_images_path(image_data_dir_path, df, label_map):
    """
        this function, will create a image path as a new column in the exisitng dataframe, and also converts the text labels into
        numerical values
        Params:
            image_data_dir_path(type: str): image directory path.
            df(type: pandas.DataFrame): Dataframe, that needed new cols.
            label_map(type; dict): Labels with their integer representation as a dictionary.
    """
    
    images_one_paths = []
    images_two_paths = []

    try: 
        for idx in range(len(df)):
            current_row = df.iloc[idx]
            id_1 = current_row["id_1"]
            id_2 = current_row["id_2"]
            extentsion_one = current_row["image_1"].split(".")[-1]
            extentsion_two = current_row["image_2"].split(".")[-1]

            image_one_path = os.path.join(image_data_dir_path, str(id_1) + f".{extentsion_one}")
            image_two_path = os.path.join(image_data_dir_path, str(id_2) + f".{extentsion_two}")

            images_one_paths.append(image_one_path)
            images_two_paths.append(image_two_path)

        df["image_1_path"] = images_one_paths
        df["image_2_path"] = images_two_paths


        df["label_idx"] = df["label"].apply(lambda x: label_map[x])
        
        return images_one_paths, images_two_paths

    except Exception as error:
        return error


def visualize(df, data_index):
    """
        this method is used to visualize the data of the entailment dataset.
        Params:
            df(type: pandas.DAtaFrame): Pandas DataFrame.
            data_index(type: int): index value in the dataframe, that needed to be visualized.
    """

    try:
        current_row = df.iloc[data_index]
        image_1 = plt.imread(current_row["image_1_path"])
        image_2 = plt.imread(current_row["image_2_path"])
        text_1 = current_row["text_1"]
        text_2 = current_row["text_2"]
        label = current_row["label"]

        plt.subplot(1, 2, 1)
        plt.imshow(image_1)
        plt.axis("off")
        plt.title("Image One")
        plt.subplot(1, 2, 2)
        plt.imshow(image_1)
        plt.axis("off")
        plt.title("Image Two")
        plt.show()

        print(f"Text one: {text_1}")
        print(f"Text two: {text_2}")
        print(f"Label: {label}")

    except Exception as error:
        return error

def get_optimizer(lr=1e-3, optimizer="adam"):
    """
        this function, used to create a keras optimer.
        Params:
            lr(type: float): Learning Rate for the optimizer.
            optimzer(type: str): optimizer type (adam or sgd)
        
        Return(type: keras.optimizer)
            this function, will return the keras optimizer.
    """
    if optimizer.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if optimizer.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.1)


def get_labels(ds):
    """
        this function, will create labels (dependent variable).
        Params:
            ds(type: tf.data.Dataset): Tensorflow dataset object.
        Return(type: np.array)
            this function, returns the labels.
    """
    labels = []
    for _, label in ds.unbatch():
        labels.append(label)
    labels = np.array(labels)
    return labels

train_labels = get_labels(train_ds)
test_labels = get_labels(test_ds)





