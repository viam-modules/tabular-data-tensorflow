import argparse
import json
import os
import sys
import typing as ty

import numpy as np
import tensorflow as tf
from keras import Model
import datetime
import tensorflow as tf
import viam
from viam.utils import create_filter
import asyncio

from viam.rpc.dial import DialOptions, Credentials
from viam.app.viam_client import ViamClient


def parse_args():
    """Dataset file and model output directory are required parameters. These must be parsed as command line
    arguments and then used as the model input and output, respectively.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", dest="data_json", type=str)
    parser.add_argument("--model_output_directory", dest="model_dir", type=str)
    args = parser.parse_args()
    return args.data_json, args.model_dir


async def connect() -> ViamClient:
    # The API key and key ID can be accessed programmatically, using the environment variable API_KEY and API_KEY_ID.
    # The user does not need to supply the API keys, they are provided automatically when the training job is submitted.
    dial_options = DialOptions.with_api_key(
        os.environ.get("API_KEY"), os.environ.get("API_KEY_ID")
    )
    return await ViamClient.create_from_dial_options(dial_options, "app.viam.com")


async def get_data_from_filter(data_client, my_filter, reading_name):
    # Store the data in a map, where the key is the date time
    data = {}
    last = None
    while True:
        tabular_data, _, last = await data_client.tabular_data_by_filter(
            my_filter, last=last
        )
        if not tabular_data:
            break
        for datum in tabular_data:
            time_received = datum.time_received
            # Truncate the time in place so it can be used for time synchronization of the data
            truncated_time = datetime.datetime(
                time_received.year,
                time_received.month,
                time_received.day,
                time_received.hour,
                time_received.minute,
                time_received.second,
            )
            data[truncated_time] = datum.data["readings"][reading_name]
    return data


def create_dataset(
    input_data,
    output_data,
    train_split,
    batch_size,
    shuffle_buffer_size,
    prefetch_buffer_size,
) -> ty.Tuple[tf.data.Dataset, tf.data.Dataset]:
    intersection_dates = set(output_data.keys())
    # Filter through the dataset and find the intersection of all the times
    # Note, that it's possible that the times don't intersect at all,
    # in which case one should employ some other technique for time synchronization
    input_keys = input_data.keys()
    for key in input_keys:
        intersection_dates.intersection(set(input_data[key].keys()))

    features = {key: [] for key in input_keys}
    labels = []

    # Update the dictionaries to only have the data belonging to the dates in the
    # intersection of all the datasets.
    for date in intersection_dates:
        for key in input_keys:
            features[key] = features[key] + [input_data[key][date]]

        labels.append(output_data[date])

    # Group together all the inputs for training
    input_tensors = {}
    for key in input_keys:
        input_tensors[key] = tf.data.Dataset.from_tensor_slices(
            np.expand_dims(np.array(features[key]), axis=-1)
        )
    inputs = tf.data.Dataset.zip((input_tensors))

    # Couple the inputs and outputs together
    output_tensor = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((inputs, output_tensor))

    # Shuffle the data for each buffer size
    # Disabling reshuffling ensures items from the training and test set will not get shuffled into each other
    dataset = dataset.shuffle(
        buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False
    )

    train_size = int(train_split * len(intersection_dates))
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    # Batch the data for multiple steps
    # If the size of training data is smaller than the batch size,
    # batch the data to expand the dimensions by a length 1 axis.
    # This will ensure that the training data is valid model input
    train_batch_size = batch_size if batch_size < train_size else train_size
    train_dataset = train_dataset.batch(train_batch_size)

    # Fetch batches in the background while the model is training.
    train_dataset = train_dataset.prefetch(prefetch_buffer_size)

    return train_dataset, test_dataset


def build_and_compile_model(norm):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,), batch_size=32),
            norm,
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
    return model


async def get_all_data_from_viam(input_names, output_name):
    # Make a ViamClient
    viam_client = await connect()
    # Instantiate a DataClient to run data client API methods on
    data_client = viam_client.data_client

    # Get all of the input data
    input_data = {}
    for name in input_names:
        input_data[name] = await get_data_from_filter(
            data_client, create_filter(component_name=name), name
        )

    # Get the output data
    output_filter = create_filter(component_name=output_name)
    output_data = await get_data_from_filter(data_client, output_filter, output_name)

    # Close ViamClient
    viam_client.close()
    return input_data, output_data

def save_model(model, model_dir):
    # Save the trained model in SavedModel format
    tf.saved_model.save(model, model_dir)

if __name__ == "__main__":
    # Set up compute device strategy
    if len(tf.config.list_physical_devices("GPU")) > 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    BATCH_SIZE = 16
    SHUFFLE_BUFFER_SIZE = 64  # Shuffle the training data by a chunk of 64 observations
    AUTOTUNE = (
        tf.data.experimental.AUTOTUNE
    )  # Adapt preprocessing and prefetching dynamically

    # Model constants
    NUM_WORKERS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
    EPOCHS = 2

    # DATA_JSON is ignored in this case but can be used if combining
    # tabular and binary data for model training.
    _, MODEL_DIR = parse_args()

    # Query and process the data from Viam so only the fields relevant to training are used
    # Provide input names, a list of sensor values that will be used to model the output value, specified by output name.
    input_data, output_data = asyncio.run(
        get_all_data_from_viam(["temperature", "humidity"], "precipitation")
    )

    # Create the datasets which includes cleaning it up to make sure they are synchronized
    train_dataset, test_dataset = create_dataset(
        input_data,
        output_data,
        train_split=0.8,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        prefetch_buffer_size=AUTOTUNE,
    )

    normalizer = tf.keras.layers.Normalization(axis=-1)
    regression_model = build_and_compile_model(normalizer)

    history = regression_model.fit(train_dataset, epochs=EPOCHS)

    save_model(regression_model, MODEL_DIR)