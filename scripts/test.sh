#!/bin/sh

python3 setup.py sdist --formats=gztar
pip install dist/model-0.1.tar.gz

API_KEY=$VIAM_DEV_API_KEY API_KEY_ID=$VIAM_DEV_API_KEY_ID python3 -m model.training --dataset_file=dataset.jsonl --model_output_directory=. --num_epochs=2
pip install pytest && pytest tests/
