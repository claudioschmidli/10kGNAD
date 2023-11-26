import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from IPython.display import clear_output



class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if "val" not in x and "lr" not in x]  # Exclude 'lr'

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(
                range(1, epoch + 2), self.metrics[metric], label=metric, ls="dashed"
            )
            val_metric_key = "val_" + metric
            if val_metric_key in self.metrics:  # Check if val_metric exists
                axs[i].plot(
                    range(1, epoch + 2),
                    self.metrics[val_metric_key],
                    label=val_metric_key,
                    ls="-",
                )

            axs[i].legend()
            axs[i].grid()
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel(metric)

        plt.tight_layout()
        plt.show()


def macro_f1_score(y_true, y_pred, threshold=0.5):
    """Calculate the Macro F1 Score for multi-label classification."""
    y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def plot_learning_schedule(EPOCHS, lrfn):
    rng = [i for i in range(EPOCHS)]
    y = [lrfn(x) for x in rng]
    plt.plot(rng, y)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.suptitle(
        "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
            y[0], max(y), y[-1]
        )
    )

def training_plot(metrics, history):
    f, ax = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric], ls="dashed")
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel(metric)
        ax[idx].plot(history.history["val_" + metric])
        ax[idx].legend([metric, "val_" + metric])
        plt.suptitle("Training curves")



