from typing import Dict

import numpy
from PIL import Image
from chainer import cuda, backend, serializers

from handwriting_embedding.models.classifier import CrossEntropyClassifier
from handwriting_embedding.models.resnet import PooledResNet


class HandwritingPredictor:
    def __init__(self, best_model_path):
        self.gpu = 0  # TODO: how to set gpu
        self.xp = cuda.cupy if int(self.gpu) >= 0 else numpy

        classes = ["alpha_num", "date", "num", "plz", "text"]  # TODO
        resnet_size = 18  # TODO: magic number
        self.base_model = PooledResNet(resnet_size)
        self.model = CrossEntropyClassifier(self.base_model, len(classes))

        if int(self.gpu) >= 0:
            backend.get_device(self.gpu).use()
            self.base_model.to_gpu()
            self.model.to_gpu()

        serializers.load_npz(best_model_path, self.model)

    def image_to_array(self, image: Image.Image):
        img_array = self.xp.array(image, dtype="float32")
        img_array /= 255
        if img_array.ndim == 2:
            img_array = self.xp.stack((img_array,) * 3, axis=-1)
        return self.xp.transpose(img_array, (2, 0, 1))

    def predict_image(self, image: Image.Image) -> Dict:
        image_array = self.image_to_array(image)
        image_batch = self.xp.expand_dims(image_array, 0)
        prediction = self.model.predict(image_batch)

        assert len(prediction) == 1
        predicted_class_id = prediction[0]
        result = {
            "predicted_class": str(predicted_class_id)  # TODO: use correct class string
        }

        return result


def main():
    image = Image.open("datasets/action_649.png")  # TODO: magic string
    predictor = HandwritingPredictor("best_models/5CHPT_softmax_model.npz")  # TODO: how to initialise model?
    prediction_result = predictor.predict_image(image)
    print(prediction_result)


if __name__ == '__main__':
    main()
