import random
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2
import tensorflow as tf
from mlu_tools.validation import validate_array_like


def from_arr(
    X,
    y=None,
    y_preds=None,
    class_names=None,
    scaling_factor=2.5,
    total_items_to_show=25,
    seed=None,
    shuffle=True,
    custom_titles=None,
    title_template="{cname}",
):
    random.seed(seed)
    if y_preds is not None:
        pred_labels = y_preds.argmax(axis=-1)
        pred_confs = y_preds[np.arange(len(pred_labels)), pred_labels]
    total_items_to_show = min(len(X), total_items_to_show)
    if shuffle:
        rand_indices = random.sample(range(len(X)), total_items_to_show)
    else:
        rand_indices = range(total_items_to_show)
    cols = int(np.ceil(np.sqrt(total_items_to_show)))
    rows = int(np.ceil(total_items_to_show / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(scaling_factor * cols, scaling_factor * rows)
    )
    img_h, img_w = X[0].shape[:2]
    true_class_y = max(img_h, img_w) * 3 / 32
    for i, rand_idx in enumerate(rand_indices):
        ax = axes[i // cols][i % cols]
        image = X[rand_idx]
        ax.imshow(image)
        if custom_titles is not None:
            ax.set_title(custom_titles[rand_idx])
        elif y is not None:
            if y_preds is not None:
                cname = (
                    class_names[pred_labels[rand_idx].item()]
                    if class_names
                    else pred_labels[rand_idx].item()
                )
                conf = pred_confs[rand_idx].item()
                title = title_template.format(cname=cname, conf=conf)
                if pred_labels[rand_idx].item() == y[rand_idx].item():
                    ax.set_title(title, color="green")
                else:
                    ax.set_title(title, color="red")
                    if class_names:
                        ax.text(
                            0,
                            true_class_y,
                            class_names[y[rand_idx].item()],
                            color="white",
                            bbox=dict(facecolor="green"),
                        )
                    else:
                        ax.text(
                            0,
                            true_class_y,
                            y[rand_idx].item(),
                            color="white",
                            bbox=dict(facecolor="green"),
                        )
            elif class_names is not None:
                ax.set_title(class_names[y[rand_idx].item()])
            else:
                ax.set_title(y[rand_idx].item())
        ax.axis("off")


def from_dir(
    X,
    y=None,
    y_preds=None,
    class_names=None,
    scaling_factor=2.5,
    total_items_to_show=25,
    seed=None,
    shuffle=True,
    custom_titles=None,
    title_template="{cname}",
):
    random.seed(seed)
    # Prioritize custom_titles over y and y_preds
    if custom_titles is not None or y is None:
        y = None
        y_preds = None

    infer_from_dir = "infer_from_dir"
    if isinstance(y, str):
        if y != infer_from_dir:
            raise Exception(
                f"'{y}' is an invalid value for y, did you mean '{infer_from_dir}'?"
            )
    elif validate_array_like(y, raise_exception=False):
        y = np.array(y)
        if y.ndim == 2 and y.shape[1] > 1:
            y = y.argmax(axis=1)

    directory_path = X

    # directory_items = os.listdir(directory_path)
    directory_items = []
    if y is infer_from_dir:
        y_temp = []
        label = 0
    for root, dirs, files in os.walk(directory_path):
        # directory_items.extend(files)
        if files:
            directory_items.extend(Path(root).iterdir())
            if y is infer_from_dir:
                y_temp.extend([label] * len(files))
                label += 1

    if y is infer_from_dir:
        y = np.array(y_temp)

    total_items_to_show = min(len(directory_items), total_items_to_show)
    if shuffle:
        rand_indices = random.sample(range(len(directory_items)), total_items_to_show)
    else:
        rand_indices = range(total_items_to_show)

    X = []
    if y is not None:
        y_temp = []
    if y_preds is not None:
        y_preds_temp = []
    # for item_name in directory_items[:total_items_to_show]:
    for rand_idx in rand_indices:
        # item_path = f"{directory_path}/{item_name}"
        item_path = directory_items[rand_idx]
        image = cv2.imread(item_path)[..., ::-1]
        X.append(image)
        if y is not None:
            y_temp.append(y[rand_idx])
        if y_preds is not None:
            y_preds_temp.append(y_preds[rand_idx])
    if y is not None:
        y = y_temp
    if y_preds is not None:
        y_preds = y_preds_temp

    from_arr(
        X,
        y,
        y_preds,
        class_names,
        scaling_factor,
        total_items_to_show,
        seed,
        shuffle=False,
        custom_titles=custom_titles,
        title_template=title_template,
    )


def from_ds(
    X,
    y=None,
    y_preds=None,
    class_names=None,
    scaling_factor=2.5,
    total_items_to_show=25,
    custom_titles=None,
    title_template="{cname}",
):
    i = 0
    image_ds = X
    X = []
    y = []
    for images, labels in image_ds:
        if i == total_items_to_show:
            break
        items_to_fetch_from_batch = min(total_items_to_show - i, len(images))
        X.extend(tf.cast(images[:items_to_fetch_from_batch], "uint8"))
        y.extend(labels[:items_to_fetch_from_batch].numpy())
        i += items_to_fetch_from_batch

    from_arr(
        X,
        y,
        y_preds,
        class_names,
        scaling_factor,
        total_items_to_show,
        shuffle=False,
        custom_titles=custom_titles,
        title_template=title_template,
    )


def from_paths(
    X,
    y=None,
    y_preds=None,
    class_names=None,
    scaling_factor=2.5,
    total_items_to_show=25,
    seed=None,
    shuffle=True,
    custom_titles=None,
    title_template="{cname}",
):
    paths = X
    X = []
    for path in paths:
        X.append(cv2.imread(path)[..., ::-1])

    from_arr(
        X,
        y,
        y_preds,
        class_names,
        scaling_factor,
        total_items_to_show,
        seed,
        shuffle=False,
        custom_titles=custom_titles,
        title_template=title_template,
    )
