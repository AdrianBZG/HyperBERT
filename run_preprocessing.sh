CONFIG_PATH="configs/preprocessing.json"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/preprocessing/preprocessing_pubmed.py --config_path=$CONFIG_PATH
python src/preprocessing/preprocessing_cora.py --config_path=$CONFIG_PATH
python src/preprocessing/preprocessing_dblp.py --config_path=$CONFIG_PATH
python src/preprocessing/preprocessing_imdb.py --config_path=$CONFIG_PATH
python src/preprocessing/preprocessing_convert_to_graphs.py