INPUT_DATA = input_data.csv
OUTPUT_DATA = text_processed_data.csv
OUTPUT_NO_OUTLIERS = data_no_outliers.csv
TRAINED_MODEL = trained_model
PREDICTIONS_FILE = predictions.csv

PYTHON = python

all: train predict

setup:
    pip install -r requirements.txt

train: data_processing.py outliers_detection.py train.py
    @echo "Training the model..."
    $(PYTHON) data_processing.py --input-file $(INPUT_DATA) --output-file $(OUTPUT_DATA)
    $(PYTHON) outliers_detection.py --input-file $(OUTPUT_DATA) --output-file $(OUTPUT_NO_OUTLIERS)
    $(PYTHON) train.py --input-file $(OUTPUT_NO_OUTLIERS) --output-model $(TRAINED_MODEL)

predict: predict.py
    @echo "Making predictions..."
    $(PYTHON) predict.py --input-file $(INPUT_DATA) --model-file $(TRAINED_MODEL) --output-file $(PREDICTIONS_FILE)

clean:
    @echo "Cleaning up..."
    rm -f $(OUTPUT_DATA) $(OUTPUT_NO_OUTLIERS) $(TRAINED_MODEL) $(PREDICTIONS_FILE)

.PHONY: all train predict clean
