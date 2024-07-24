import argparse
import json
import os
import sys
import typing as ty

import tensorflow as tf
from keras import Model

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
    # The API key and key ID can be accessed programmatically, using the environment variable API_KEY and API_KEY_ID 
    dial_options = DialOptions.with_api_key(os.environ.get('API_KEY'), os.environ.get('API_KEY_ID'))
    return await ViamClient.create_from_dial_options(dial_options, "app.viam.com")

async def get_data_from_filter(data_client, my_filter, reading_name):
    # Store the data in a map, where the key is the date time
    data = {}
    last = None
    while True:
        tabular_data, _, last = await data_client.tabular_data_by_filter(my_filter, last=last)
        if not tabular_data:
            break
        for datum in tabular_data:
            data[datum.time_received] = datum.data["readings"][reading_name]
    return data

def create_dataset(input_data, output_data):
    # Filter through the dataset and get data for every second
    return 
        
def build_and_compile_model(norm):
  model = tf.keras.Sequential([
      norm,
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

async def get_all_data_from_viam():
    # Make a ViamClient
    viam_client = await connect()
    # Instantiate a DataClient to run data client API methods on
    data_client = viam_client.data_client

    # Get all of the input data
    input_data = {}
    input_data["temperature"] = await get_data_from_filter(data_client, create_filter(component_name="temperature"), "temperature") 
    input_data["humidity"] = await get_data_from_filter(data_client, create_filter(component_name="humidity"), "humidity")

    # Get the output data
    output_filter = create_filter(component_name="precipitation")
    output_data = await get_data_from_filter(data_client, output_filter, "precipitation")

    # Close ViamClient 
    viam_client.close()
    return input_data, output_data

if __name__ == '__main__':
    # Set up compute device strategy
    if len(tf.config.list_physical_devices("GPU")) > 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    BATCH_SIZE = 16
    # TARGET_SHAPE is the intended shape of the model after resizing
    # For EfficientNet, this must be some multiple of 128 according to the documentation.
    TARGET_SHAPE = (384, 384, 3)
    SHUFFLE_BUFFER_SIZE = (
        64  # Shuffle the training data by a chunk of 64 observations
    )
    AUTOTUNE = (
        tf.data.experimental.AUTOTUNE
    )  # Adapt preprocessing and prefetching dynamically

    # Model constants
    NUM_WORKERS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
    EPOCHS = 2

    DATA_JSON, MODEL_DIR = parse_args()

    input_data, output_data = asyncio.run(get_all_data_from_viam())

    # Create the datasets which includes cleaning it up to make sure they are synchronized
    train_features, train_labels, test_features, test_labels = create_dataset(input_data, output_data)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    regression_model = build_and_compile_model(normalizer)

    history = regression_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        epochs=100)


