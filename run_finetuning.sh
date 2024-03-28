CONFIG_PATH="configs/finetuning.json"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/finetuning.py --config_path=$CONFIG_PATH