import os 

def get_data(dataframe_url, image_url):
    try:
        os.system(f"wget {dataframe_url} --quiet")
        os.system(f"wget {image_url} --quiet")
        os.system("tar -xzf tweet_images.tar.gz")

    except Exception as error:
        return error


def create_images_path(image_data_dir_path, df, label_map):
    
    images_one_paths = []
    images_two_paths = []

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


def visualize(df, data_index):
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
    "Select optmizer between adam and sgd with momentum"
    if optimizer.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if optimizer.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.1)


def get_labels(ds):
    labels = []
    for _, label in ds.unbatch():
        labels.append(label)
    labels = np.array(labels)
    return labels

train_labels = get_labels(train_ds)
test_labels = get_labels(test_ds)





