import tensorflow as tf


DepthNet = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),

    # Block 1
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu", depth_multiplier=2),
    tf.keras.layers.Dropout(0.1),

    # Block 2
    tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    # Block 3
    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu", depth_multiplier=2),
    tf.keras.layers.Dropout(0.3),

    # Block 4
    tf.keras.layers.Conv2D(256, 3, activation="relu", strides=2, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu"),
    tf.keras.layers.Dropout(0.3),

    # Block 5
    tf.keras.layers.Conv2D(512, 3, activation="relu", strides=2, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    # Dense Layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])


def all_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = tf.keras.Sequential()

    # First convolution block
    model.add(tf.keras.layers.Conv2D(96, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(96, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    # Second convolution block
    model.add(tf.keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    # Third convolution block
    model.add(tf.keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    # Fully Convolutional Layers (without fully connected layers)
    model.add(tf.keras.layers.Conv2D(10, kernel_size=1, activation='softmax'))

    # Flatten and apply softmax
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    return model
