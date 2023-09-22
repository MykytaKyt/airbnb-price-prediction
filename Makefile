all: data_process text_process new_features outliers

data_process:
	python data_processing.py --input-file data/train.csv --output-file data/prep.csv
text_process:
	python text_processing.py --input-file data/prep.csv --output-file data/text_p.csv
new_features:
	python feature_engineering.py --input-file data/text_p.csv --output-file data/inpuded.csv
outliers:
	python outliers_detection.py --input-file data/inpuded.csv --output-file data/outlier_removed.csv
train:
	python train.py --input-file data/outlier_removed.csv