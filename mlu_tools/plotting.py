import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from .validation import validate_array_like


def grid_plot(X, y=None, y_preds=None, class_names=None, scaling_factor=2.5, total_items_to_show=25, seed=None):
    random.seed(seed)
    if y is not None:
        y = np.array(y, dtype=int)
        if y.ndim == 2 and y.shape[1] > 1:
            y = y.argmax(axis=1)

    if y_preds is not None:
        y_preds = np.array(y_preds, dtype=int)
        if y_preds.ndim == 2 and y_preds.shape[1] > 1:
            y_preds = y_preds.argmax(axis=1)

    if isinstance(X, str):
        directory_path = X
        directory_items = os.listdir(directory_path)
        total_items_to_show = min(len(directory_items), total_items_to_show)
        rand_indices = random.sample(range(len(directory_items)), total_items_to_show)
        X = []
        # for item_name in directory_items[:total_items_to_show]:
        for rand_idx in rand_indices:
            # item_path = f"{directory_path}/{item_name}"
            item_path = f"{directory_path}/{directory_items[rand_idx]}"
            image = cv2.imread(item_path)[..., ::-1]
            X.append(image)                                          
    # elif not isinstance(X, (list, np.ndarray, tuple)):
    #     raise(Exception("Either provide an array of images or \
    #                     a path to a directory containing images as the first argument."))
    else:
        custom_message = ("Either provide an array of images or "
                            "a path to a directory containing images as the first argument.")
        validate_array_like(X, arg_name="X", custom_message=custom_message)

    # X = X[:total_items_to_show]
    total_items_to_show = min(len(X), total_items_to_show)
    rand_indices = random.sample(range(len(X)), total_items_to_show)
    cols = int(np.ceil(np.sqrt(total_items_to_show)))
    rows = int(np.ceil(total_items_to_show/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(scaling_factor*cols, scaling_factor*rows))
    for i, rand_idx in enumerate(rand_indices):
        ax = axes[i//cols][i%cols]
        image = X[rand_idx]
        ax.imshow(image)
        if y is not None:
            if y_preds is not None:
                if y_preds[rand_idx].item() == y[rand_idx].item():
                    ax.set_title(class_names[y[rand_idx].item()], color="green")
                else:
                    ax.set_title(class_names[y_preds[rand_idx].item()], color="red")
                    ax.text(0, 2, class_names[y[rand_idx].item()], color='white', bbox=dict(facecolor='green'))
            else:
                ax.set_title(class_names[y[rand_idx].item()])
        ax.axis("off")


def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    
    epochs = range(len(metric_value_1))

    plt.plot(epochs, metric_value_1, "blue", label=metric_name_1)
    plt.plot(epochs, metric_value_2, "red", label=metric_name_2)

    plt.title(str(plot_name))

    plt.legend()
