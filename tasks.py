import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import celery
from PIL import Image
from celery import Celery

from handwriting_embedding.predict import HandwritingClassifier


class ClassificationTask(celery.Task):

    def __init__(self):
        sys.path.append(str(Path(__file__).resolve().parent))
        self.config = {
            'model_config_path': os.environ.get('CLASSIFICATION_CONFIG_PATH', None),
            'device_id': int(os.environ.get('CLASSIFICATION_DEVICE', -1))
        }
        print(self.config)
        assert self.config['model_config_path'] is not None, "You must supply a path to a model configuration in the " \
                                                             "environment variable CLASSIFICATION_CONFIG_PATH "
        self.handwriting_classifier = None

    def initialize(self):
        if self.handwriting_classifier is not None:
            return
        self.handwriting_classifier = HandwritingClassifier(self.config["model_config_path"], self.config["device_id"])


broker_address = os.environ.get('BROKER_ADDRESS', 'localhost')
app = Celery('wpi_demo', backend='rpc://', broker=f"pyamqp://guest@{broker_address}//")


@app.task(name='handwriting_classification', base=ClassificationTask)
def classify(task_data):
    classify.initialize()
    image = base64.b85decode(task_data['image'])
    io = BytesIO(image)
    io.seek(0)

    with Image.open(io) as decoded_image:
        decoded_image = decoded_image.convert('RGB')
        classification_result = classify.handwriting_classifier.predict_image(decoded_image)

    return json.dumps(classification_result)
