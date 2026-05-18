import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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

from config import BEST_PARAMS, MODEL_BASE_PARAMS

tf.get_logger().setLevel("ERROR")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except AttributeError:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def normalize_best_params(params=None):
    normalized = dict(BEST_PARAMS)
    if params:
        normalized.update(dict(params))
    for key in MODEL_BASE_PARAMS:
        normalized[key] = int(normalized[key])
        if normalized[key] <= 0:
            raise ValueError("{} must be positive.".format(key))
    normalized["dropout"] = float(normalized["dropout"])
    normalized["lr"] = float(normalized["lr"])
    return normalized


def smape_metric_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def mse_mae_loss(mse_weight=5.0, mae_weight=1.0):
    def loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        mae = tf.reduce_mean(tf.abs(y_pred - y_true))
        return (float(mse_weight) * mse) + (float(mae_weight) * mae)
    return loss


def compile_model(model, params):
    loss_type = str(params.get("loss_type", "mae")).lower()
    if loss_type == "mae":
        loss = "mae"
    elif loss_type == "mse_mae":
        loss = mse_mae_loss(
            mse_weight=float(params.get("loss_mse_weight", 5.0)),
            mae_weight=float(params.get("loss_mae_weight", 1.0)),
        )
    else:
        raise ValueError("Final pipeline keeps only mae and mse_mae loss flows. Got {}".format(loss_type))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(params["lr"])),
        loss=loss,
        metrics=["mse", "mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def build_x0_encoder(input_shape0, params):
    input_ = Input(shape=input_shape0, name="x0_input")

    def dense_block(tensor):
        tensor = Dense(params["x0_dense1"], activation="relu")(tensor)
        tensor = Dense(params["x0_dense2"], activation="relu")(tensor)
        tensor = Dense(params["x0_dense3"], activation="relu")(tensor)
        tensor = Dense(params["x0_dense4"], activation="relu")(tensor)
        return tensor

    left = dense_block(input_)
    right = dense_block(input_)
    merged = Dropout(params["dropout"])(Concatenate()([left, right]))

    def conv_block(tensor):
        tensor = Conv1D(params["x0_conv1"], 3, activation="relu", padding="same")(tensor)
        tensor = MaxPool1D()(tensor)
        tensor = Conv1D(params["x0_conv2"], 3, activation="relu", padding="same")(tensor)
        tensor = MaxPool1D()(tensor)
        tensor = Conv1D(params["x0_conv3"], 3, activation="relu", padding="same")(tensor)
        tensor = Conv1D(params["x0_conv4"], 3, activation="relu", padding="same")(tensor)
        gap = GlobalAveragePooling1D()(tensor)
        attention = Dense(params["x0_attn_hidden"], activation="relu")(gap)
        attention = Dense(params["x0_attn_channels"], activation="sigmoid")(attention)
        attended_input = tensor
        if params["x0_attn_channels"] != params["x0_conv4"]:
            attended_input = Conv1D(params["x0_attn_channels"], 1, activation="relu", padding="same")(attended_input)
        attended = Multiply()([attended_input, Reshape((1, params["x0_attn_channels"]))(attention)])
        attended = Dense(params["x0_proj"], activation="relu")(attended)
        return Concatenate()([tensor, attended])

    left = conv_block(merged)
    right = conv_block(merged)
    output = GlobalAveragePooling1D()(Concatenate()([left, right]))
    return Model(inputs=input_, outputs=output, name="x0_encoder")


def build_x1_encoder(input_shape1, params):
    input_ = Input(shape=input_shape1, name="x1_input")

    def conv_block(tensor):
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
            attended_input = Conv2D(params["x1_attn_channels"], (1, 1), activation="relu", padding="same")(attended_input)
        attended = Multiply()([attended_input, Reshape((1, 1, params["x1_attn_channels"]))(attention)])
        attended = Dense(params["x1_proj"], activation="relu")(attended)
        return Concatenate()([tensor, attended])

    left = conv_block(input_)
    right = conv_block(input_)
    output = GlobalAveragePooling2D()(Concatenate()([left, right]))
    return Model(inputs=input_, outputs=output, name="x1_encoder")


def build_model(input_shape0, input_shape1, best_params=None):
    params = normalize_best_params(best_params)
    x0_model = build_x0_encoder(input_shape0, params)
    x1_model = build_x1_encoder(input_shape1, params)
    concatenated = Concatenate(name="x0x1_concat")([x0_model.output, x1_model.output])
    gate = Dense(int(concatenated.shape[-1]), activation="sigmoid", name="fusion_gate")(concatenated)
    fused = Multiply(name="gated_features")([concatenated, gate])
    output = Dense(1, name="regression_output")(fused)
    return compile_model(Model(inputs=[x0_model.input, x1_model.input], outputs=output, name="x0x1_gated_fusion"), params)
