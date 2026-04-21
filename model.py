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

from config import BEST_PARAMS


INTEGER_PARAM_KEYS = tuple(key for key in BEST_PARAMS if key not in {"dropout", "lr"})


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalize_best_params(best_params=None):
    raw_params = BEST_PARAMS.copy() if best_params is None else dict(best_params)
    normalized = BEST_PARAMS.copy()
    normalized.update(raw_params)

    for key in INTEGER_PARAM_KEYS:
        normalized[key] = int(normalized[key])
        if normalized[key] <= 0:
            raise ValueError("{} must be a positive integer.".format(key))

    normalized["dropout"] = float(normalized["dropout"])
    normalized["lr"] = float(normalized["lr"])
    return normalized


def smape_metric_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def smape_loss_tf(y_true, y_pred, eps=1e-6):
    numerator = tf.abs(y_pred - y_true)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0 + eps
    return tf.reduce_mean(numerator / denominator)


def huber_smape_loss(alpha=0.05, delta=1.0):
    huber_fn = tf.keras.losses.Huber(delta=delta)

    def loss(y_true, y_pred):
        huber = huber_fn(y_true, y_pred)
        smape = smape_loss_tf(y_true, y_pred) / 100.0
        return huber + (alpha * smape)

    return loss


def build_model(input_shape0, input_shape1, best_params=None):
    params = normalize_best_params(best_params)

    def build_x0_branch(input_shape):
        x = input_ = Input(shape=input_shape)

        def dense_feature_block(tensor):
            tensor = Dense(units=params["x0_dense1"], activation="relu")(tensor)
            tensor = Dense(units=params["x0_dense2"], activation="relu")(tensor)
            tensor = Dense(units=params["x0_dense3"], activation="relu")(tensor)
            tensor = Dense(units=params["x0_dense4"], activation="relu")(tensor)
            return tensor

        left = dense_feature_block(x)
        right = dense_feature_block(x)
        merged = Concatenate()([left, right])
        merged = Dropout(params["dropout"])(merged)

        def conv_feature_block(tensor):
            tensor = Conv1D(filters=params["x0_conv1"], kernel_size=3, activation="relu", padding="same")(tensor)
            tensor = MaxPool1D()(tensor)
            tensor = Conv1D(filters=params["x0_conv2"], kernel_size=3, activation="relu", padding="same")(tensor)
            tensor = MaxPool1D()(tensor)
            tensor = Conv1D(filters=params["x0_conv3"], kernel_size=3, activation="relu", padding="same")(tensor)
            tensor = Conv1D(filters=params["x0_conv4"], kernel_size=3, activation="relu", padding="same")(tensor)

            gap = GlobalAveragePooling1D()(tensor)
            attention = Dense(params["x0_attn_hidden"], activation="relu")(gap)
            attention = Dense(params["x0_attn_channels"], activation="sigmoid")(attention)

            attended_input = tensor
            if params["x0_attn_channels"] != params["x0_conv4"]:
                attended_input = Conv1D(
                    filters=params["x0_attn_channels"],
                    kernel_size=1,
                    activation="relu",
                    padding="same",
                )(attended_input)

            attended = Multiply()([attended_input, Reshape((1, params["x0_attn_channels"]))(attention)])
            attended = Dense(units=params["x0_proj"], activation="relu")(attended)
            return Concatenate()([tensor, attended])

        left = conv_feature_block(merged)
        right = conv_feature_block(merged)
        output = Concatenate()([left, right])
        output = GlobalAveragePooling1D()(output)
        return Model(inputs=input_, outputs=output)

    def build_x1_branch(input_shape):
        x = input_ = Input(shape=input_shape)

        def conv_feature_block(tensor):
            tensor = Conv2D(params["x1_conv1"], (3, 3), activation="relu", padding="same")(tensor)
            tensor = MaxPool2D()(tensor)
            tensor = Conv2D(params["x1_conv2"], (3, 3), activation="relu", padding="same")(tensor)
            tensor = MaxPool2D()(tensor)
            tensor = Conv2D(params["x1_conv3"], (3, 3), activation="relu", padding="same")(tensor)
            tensor = Conv2D(params["x1_conv4"], (3, 3), activation="relu", padding="same")(tensor)

            gap = GlobalAveragePooling2D()(tensor)
            attention = Dense(params["x1_attn_hidden"], activation="relu")(gap)
            attention = Dense(params["x1_attn_channels"], activation="sigmoid")(attention)

            attended_input = tensor
            if params["x1_attn_channels"] != params["x1_conv4"]:
                attended_input = Conv2D(
                    params["x1_attn_channels"],
                    (1, 1),
                    activation="relu",
                    padding="same",
                )(attended_input)

            attended = Multiply()([attended_input, Reshape((1, 1, params["x1_attn_channels"]))(attention)])
            attended = Dense(units=params["x1_proj"], activation="relu")(attended)
            return Concatenate()([tensor, attended])

        left = conv_feature_block(x)
        right = conv_feature_block(x)
        output = Concatenate()([left, right])
        output = GlobalAveragePooling2D()(output)
        return Model(inputs=input_, outputs=output)

    def gated_fusion(x0_features, x1_features):
        concatenated = Concatenate()([x0_features, x1_features])
        gate = Dense(int(concatenated.shape[-1]), activation="sigmoid")(concatenated)
        return Multiply()([concatenated, gate])

    x0_model = build_x0_branch(input_shape0)
    x1_model = build_x1_branch(input_shape1)
    fused = gated_fusion(x0_model.output, x1_model.output)
    output = Dense(units=1)(fused)

    model = Model(inputs=[x0_model.input, x1_model.input], outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=params["lr"]),
        loss=huber_smape_loss(alpha=0.05, delta=1.0),
        metrics=["mse", "mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model
