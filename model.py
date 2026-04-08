import os
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    MaxPool1D,
    MaxPool2D,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)
    except Exception:
        tf.random.set_seed(seed)


def smape_loss_tf(y_true, y_pred, eps=1e-6):
    numerator = tf.abs(y_pred - y_true)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0 + eps
    return tf.reduce_mean(numerator / denominator)


def huber_smape_loss(alpha=0.05, delta=1.0):
    huber_fn = tf.keras.losses.Huber(delta=delta)

    def loss(y_true, y_pred):
        huber = huber_fn(y_true, y_pred)
        smape = smape_loss_tf(y_true, y_pred) / 100.0
        return huber + alpha * smape

    return loss


def build_multimodal_model(input_shape0, input_shape1, params):
    filters1 = params["filters1"]
    filters2 = params["filters2"]
    filters3 = params["filters3"]
    filters4 = params["filters4"]
    filters5 = params["filters5"]
    filters6 = params["filters6"]
    dropout = params["dropout"]
    lr = params["lr"]

    def x0_branch(input_shape):
        x = input_tensor = Input(shape=input_shape)

        def dense_stack(tensor):
            tensor = Dense(units=filters1, activation="relu")(tensor)
            tensor = Dense(units=filters2, activation="relu")(tensor)
            tensor = Dense(units=filters2, activation="relu")(tensor)
            tensor = Dense(units=filters3, activation="relu")(tensor)
            return tensor

        x0 = dense_stack(x)
        x1 = dense_stack(x)
        merged = Concatenate()([x0, x1])
        merged = Dropout(dropout)(merged)

        def temporal_stack(tensor):
            tensor = Conv1D(filters=filters4, kernel_size=3, activation="relu", padding="same")(tensor)
            tensor = MaxPool1D()(tensor)
            tensor = Conv1D(filters=filters5, kernel_size=3, activation="relu", padding="same")(tensor)
            tensor = MaxPool1D()(tensor)
            tensor = Conv1D(filters=filters6, kernel_size=3, activation="relu", padding="same")(tensor)
            tensor = Conv1D(filters=filters6, kernel_size=3, activation="relu", padding="same")(tensor)

            gap = GlobalAveragePooling1D()(tensor)
            dense = Dense(max(filters6 // 8, 4), activation="relu")(gap)
            dense = Dense(filters6, activation="sigmoid")(dense)
            attention = Multiply()([tensor, Reshape((1, filters6))(dense)])
            attention = Dense(units=filters4, activation="relu")(attention)
            return Concatenate()([tensor, attention])

        x0 = temporal_stack(merged)
        x1 = temporal_stack(merged)
        output = Concatenate()([x0, x1])
        output = GlobalAveragePooling1D()(output)
        return Model(inputs=input_tensor, outputs=output)

    def x1_branch(input_shape):
        x = input_tensor = Input(shape=input_shape)

        def conv_stack(tensor):
            tensor = Conv2D(filters1, (3, 3), activation="relu", padding="same")(tensor)
            tensor = MaxPool2D()(tensor)
            tensor = Conv2D(filters2, (3, 3), activation="relu", padding="same")(tensor)
            tensor = MaxPool2D()(tensor)
            tensor = Conv2D(filters3, (3, 3), activation="relu", padding="same")(tensor)
            tensor = Conv2D(filters3, (3, 3), activation="relu", padding="same")(tensor)

            gap = GlobalAveragePooling2D()(tensor)
            dense = Dense(max(filters3 // 8, 4), activation="relu")(gap)
            dense = Dense(filters3, activation="sigmoid")(dense)
            attention = Multiply()([tensor, Reshape((1, 1, filters3))(dense)])
            attention = Dense(units=filters4, activation="relu")(attention)
            return Concatenate()([tensor, attention])

        x0 = conv_stack(x)
        x1 = conv_stack(x)
        output = Concatenate()([x0, x1])
        output = GlobalAveragePooling2D()(output)
        return Model(inputs=input_tensor, outputs=output)

    def gated_fusion(x0_feat, x1_feat):
        merged = Concatenate()([x0_feat, x1_feat])
        gate = Dense(int(merged.shape[-1]), activation="sigmoid")(merged)
        return Multiply()([merged, gate])

    x0_model = x0_branch(input_shape0)
    x1_model = x1_branch(input_shape1)

    fused = gated_fusion(x0_model.output, x1_model.output)
    output = Dense(units=1)(fused)

    model = Model(inputs=[x0_model.input, x1_model.input], outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=huber_smape_loss(alpha=0.05, delta=1.0),
        metrics=["mse", "mae"],
    )
    return model
