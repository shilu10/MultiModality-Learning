import tensorflow.keras as keras 
import tensorflow as tf 


class ManualTrainer: 
    
    def __init__(self):
        pass 

    def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
        with tf.GradientTape() as tape:
            logits = model(train_ds, training=True)
            loss = loss_fn(y, logits)
        
        model_params = model.trainable_weights
        grads = tape.gradient(loss_value, model_params)
        optimizer.apply_gradients(zip(grads, model_params))
        train_acc_metric.update_state(y, logits)

        return loss_value

    def test_step(x, y, model, loss_fn, val_acc_metric):
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)
        val_acc_metric.update_state(y, val_logits)

        return loss_value

    def manual_train(train_dataset,
            val_dataset, 
            model,
            optimizer,
            loss_fn,
            train_acc_metric,
            val_acc_metric,
            epochs=10, 
            log_step=200, 
            val_log_step=50):
    
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            train_loss = []   
            val_loss = []

            # Iterate over the batches of the dataset
            for step, (x_batch_train, y_batch_train) in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
                loss_value = train_step(x_batch_train, y_batch_train, 
                                        model, optimizer, 
                                        loss_fn, train_acc_metric)
                train_loss.append(float(loss_value))

            # Run a validation loop at the end of each epoch
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                val_loss_value = test_step(x_batch_val, y_batch_val, 
                                        model, loss_fn, 
                                        val_acc_metric)
                val_loss.append(float(val_loss_value))
                
            # Display metrics at the end of each epoch
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            val_acc = val_acc_metric.result()
            print("Validation acc: %.4f" % (float(val_acc),))

            # Reset metrics at the end of each epoch
            train_acc_metric.reset_states()
            val_acc_metric.reset_states()

            # 3️⃣ log metrics using wandb.log
            wandb.log({'epochs': epoch,
                    'loss': np.mean(train_loss),
                    'acc': float(train_acc), 
                    'val_loss': np.mean(val_loss),
                    'val_acc':float(val_acc)})