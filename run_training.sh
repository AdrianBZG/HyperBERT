CONFIG_PATH="configs/training.json"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/main.py --config_path=$CONFIG_PATH