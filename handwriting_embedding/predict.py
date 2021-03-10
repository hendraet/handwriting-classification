import json

import numpy
from PIL import Image
from chainer import backend, serializers, cuda
from chainer.backends.cuda import GpuDevice

from handwriting_embedding.models.classifier import CrossEntropyClassifier
from handwriting_embedding.models.resnet import PooledResNet


class HandwritingPredictor:
    def __init__(self, prediction_config_path="prediction_config.json"):
        self.gpu = 0  # TODO: how to set gpu

        with open(prediction_config_path) as prediction_config_file:
            prediction_config = json.load(prediction_config_file)

        classes = sorted(prediction_config["classes"])
        self.idx_to_label_map = {i: label for i, label in enumerate(classes)}
        self.base_model = PooledResNet(prediction_config["resnet_size"])
        self.model = CrossEntropyClassifier(self.base_model, len(classes))

        if int(self.gpu) >= 0:
            backend.get_device(self.gpu).use()
            self.base_model.to_gpu()
            self.model.to_gpu()

        serializers.load_npz(prediction_config["model_path"], self.model)

    @staticmethod
    def image_to_array(image, xp):
        img_array = xp.array(image, dtype="float32")
        img_array /= 255
        if img_array.ndim == 2:
            img_array = xp.stack((img_array,) * 3, axis=-1)
        return xp.transpose(img_array, (2, 0, 1))

    def predict_image(self, image):
        xp = cuda.cupy if isinstance(self.model.device, GpuDevice) else numpy
        image_array = self.image_to_array(image, xp)
        image_batch = xp.expand_dims(image_array, 0)
        prediction = self.model.predict(image_batch)

        assert len(prediction) == 1
        predicted_class_id = int(prediction[0])
        result = {
            "predicted_class": self.idx_to_label_map[predicted_class_id]
        }

        return result


def debug():
    image_path = "datasets/action_649.png"
    image = Image.open(image_path)

    predictor = HandwritingPredictor()
    prediction_result = predictor.predict_image(image)
    print(prediction_result)


if __name__ == '__main__':
    debug()
