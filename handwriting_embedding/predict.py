import json

import chainer
import numpy
from chainer import cuda
from chainer.backends.cuda import GpuDevice

from handwriting_embedding.dataset_utils import image_to_array
from handwriting_embedding.models.classifier import CrossEntropyClassifier
from handwriting_embedding.models.resnet import PooledResNet
from prep.image_processing.binarise_imgs import binarise_pil_image
from prep.image_processing.resize_images import resize_img


class HandwritingClassifier:
    def __init__(self, prediction_config_path="prediction_config.json", gpu=-1):
        self.gpu = gpu

        with open(prediction_config_path) as prediction_config_file:
            prediction_config = json.load(prediction_config_file)

        classes = sorted(prediction_config["classes"])
        long_class_label_dict = {
            "alpha_num": "Alphanumeric",
            "alphanum": "Alphanumeric",
            "date": "Date",
            "num": "Number",
            "plz": "Zip Code",
            "text": "Word"
        }
        self.idx_to_label_map = {i: long_class_label_dict[label] for i, label in enumerate(classes)}

        self.input_image_size = prediction_config["input_image_size"]
        self.base_model = PooledResNet(prediction_config["resnet_size"])
        self.model = CrossEntropyClassifier(self.base_model, len(classes))

        with numpy.load(prediction_config["model_path"]) as f:
            chainer.serializers.NpzDeserializer(f, strict=True).load(self.model)

        if int(self.gpu) >= 0:
            with chainer.using_device(chainer.get_device(self.gpu)):
                self.base_model.to_device(self.gpu)
                self.model.to_device(self.gpu)

    def preprocess_image(self, image):
        greyscale_image = image.convert("L")
        binarised_image = binarise_pil_image(greyscale_image)
        resized_image = resize_img(binarised_image, self.input_image_size, padding_color=255)
        return resized_image

    def predict_image(self, image):
        preprocessed_image = self.preprocess_image(image)

        xp = cuda.cupy if isinstance(self.model.device, GpuDevice) else numpy
        image_array = image_to_array(preprocessed_image, invert_colours=True)

        image_array = xp.array(image_array)
        image_batch = xp.expand_dims(image_array, 0)
        prediction, confidence = self.model.predict(image_batch, return_confidence=True)
        confidence = chainer.backends.cuda.to_cpu(confidence)

        assert len(prediction) == 1 and len(confidence) == 1
        predicted_class_id = int(prediction[0])
        result = {
            "predicted_class": self.idx_to_label_map[predicted_class_id],
            "confidence": float(confidence[0])
        }

        return result
