PYTHON_VERSION=python3.9
VENV_DIR=.venv
MODEL_DIR=$(HOME)/models

# Setup a virtual environment, install dependencies, and download models
setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON_VERSION) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip setuptools wheel
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install lap==0.4.0
	$(VENV_DIR)/bin/pip install -r requirements.txt
	mkdir -p $(MODEL_DIR)
	wget -O $(MODEL_DIR)/yolov8l-face.pt https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt
	wget -O $(MODEL_DIR)/yolov8l.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt
	wget -O $(MODEL_DIR)/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
	$(VENV_DIR)/bin/gdown --id 1J-PDWDAkYCdT8T2Nxn3Q_-iOHH_t-9YP -O $(MODEL_DIR)/pretrain_TalkSet.model
	touch $(VENV_DIR)/bin/activate

# Clean the environment
clean:
	rm -rf $(VENV_DIR)
	rm -rf ~/.models
	rm -rf ~/.cache/models