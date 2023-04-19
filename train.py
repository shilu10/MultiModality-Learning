from focal_loss import SparseCategoricalFocalLoss
import tensorflow_addons as tfa
import tensorflow as tf 
import tensorflow.keras as keras 


def train_model(multimodal_model, epochs, lr, batch_size, optimizer, train_labels, train_ds, test_ds): 
    wandb_callbacks = [
        WandbMetricsLogger(log_freq=5),
        WandbModelCheckpoint(filepath="my_model_{epoch:02d}")
    ]


    train_labels_ohe = keras.utils.to_categorical(train_labels)
    class_totals = train_labels_ohe.sum(axis=0)
    class_weight = dict()

    for i in range(0, len(class_totals)):
        class_weight[i] = class_totals.max() / class_totals[i]
    
    opt = get_optimizer(lr, optimizer)
    
    multimodal_model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", 
                                             metrics="accuracy")
    multimodal_model.fit(train_ds, epochs=epochs, 
                                       validation_data=test_ds, batch_size=batch_size, 
                                       callbacks=wandb_callbacks, class_weight=class_weight)