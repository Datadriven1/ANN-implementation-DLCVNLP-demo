import tensorflow as tf
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    LAYERS = [tf.keras.layers.Flatten(input_shape = [28,28], name = "inputLayer"),
          tf.keras.layers.Dense(300, activation = "relu", name = "hideenLayer1"),
          tf.keras.layers.Dense(100, activation = "relu", name = "hideenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation = "softmax", name = "outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile (loss = LOSS_FUNCTION, optimizer= OPTIMIZER, metrics = METRICS)

    return model_clf ## <<< untrained model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def save_plot(history, plot, plot_dir_path):
    def _create_base_plot(history):
        pd.DataFrame(history.history).plot(figsize = (10, 7))
        plt.grid(True)
        unique_filename = get_unique_filename(plot)
        plotPath = os.path.join(plot_dir_path, unique_filename)
        plt.savefig(plotPath)
        plt.show()

    _create_base_plot(history)

def CALLBACKS(log_dir=None):
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    ## EARLY_STOPPING_CALLBACK

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    ## Model Checkpointing callback

    CKPT_path = "model_ckpt.h5"

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, checkpointing_cb]
    
    return CALLBACKS_LIST
