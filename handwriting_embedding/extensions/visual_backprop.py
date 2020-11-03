from collections import namedtuple

import chainer
import chainer.functions as F
import numpy as np
from PIL import Image
from chainer import cuda
from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.pooling.pooling_2d import Pooling2D
from chainer.training import Extension

Size = namedtuple('Size', ['height', 'width'])


class VisualBackprop(Extension):

    def __init__(self, image, label, predictor, visualization_anchors, xp):
        super().__init__()
        self.image = image
        self.label = label
        self.image_size = Size(height=image.shape[-2], width=image.shape[-1])
        self.predictor = predictor
        self.visualization_anchors = visualization_anchors
        self.xp = xp

    def __call__(self, trainer):
        batch = self.xp.array(np.expand_dims(self.image, 0))
        self.predictor(batch)
        backprop_visualizations = self.get_backprop_visualizations()
        image = self.compose_image_and_visual_backprop(self.image, backprop_visualizations[0][0])

        epoch = trainer.updater.epoch
        filename = f"result/visual_backprop_{self.label}_{epoch}.png"
        with open(filename, "wb") as out_img:
            image.save(out_img)

    def get_backprop_visualizations(self):
        backprop_visualizations = []
        for visanchor in self.visualization_anchors:
            vis_targets = self.predictor
            for target in visanchor:
                vis_targets = getattr(vis_targets, target)
            if vis_targets is not None:
                if not hasattr(vis_targets, '__iter__'):
                    vis_targets = [vis_targets]
                for vis_target in vis_targets:
                    backprop_visualizations.append(self.perform_visual_backprop(vis_target))

        return backprop_visualizations

    def array_to_image(self, array):
        if array.shape[0] == 1:
           # image is black and white, we need to trick the system into thinking, that we are having an RGB image
            array = self.xp.tile(array, (3, 1, 1))
        array = array.copy() * 255
        return Image.fromarray(cuda.to_cpu(array.transpose(1, 2, 0)).astype(np.uint8), "RGB").convert("RGBA")

    def compose_image_and_visual_backprop(self, original_image, backprop_image):
        backprop_image = self.array_to_image(
            self.xp.tile(backprop_image, (3, 1, 1))
        ).resize(
            (self.image_size.width, self.image_size.height)
        )
        # original_image = self.array_to_image(original_image)
        original_image = self.array_to_image(np.zeros_like(original_image))
        original_image = original_image.convert("RGBA")
        backprop_image = backprop_image.convert("RGBA")


        resulting_image = Image.blend(original_image, backprop_image, 0.6)
        return resulting_image

    def traverse_computational_graph(self, node, feature_map):
        if isinstance(node, Convolution2DFunction) or isinstance(node, Pooling2D):
            feature_map = self.scale_layer(feature_map, node)

        if node.inputs[0].creator is None:
            return feature_map
        return self.traverse_computational_graph(node.inputs[0].creator, feature_map)

    def scale_layer(self, feature_map, node):
        input_data = node.inputs[0].data
        _, _, in_height, in_width = input_data.shape
        _, _, feature_height, feature_width = feature_map.shape
        kernel_height = in_height + 2 * node.ph - node.sy * (feature_height - 1)
        kernel_width = in_width + 2 * node.pw - node.sx * (feature_width - 1)
        scaled_feature = F.deconvolution_2d(
            feature_map,
            self.xp.ones((1, 1, kernel_height, kernel_width)),
            stride=(node.sy, node.sx),
            pad=(node.ph, node.pw),
            outsize=(in_height, in_width),
        )
        averaged_feature_map = F.average(input_data, axis=1, keepdims=True)
        feature_map = scaled_feature * averaged_feature_map
        return feature_map

    def perform_visual_backprop(self, variable):
        with chainer.no_backprop_mode(), chainer.cuda.get_device_from_array(variable.data):
            # self.xp = cuda.get_array_module(variable)
            averaged_feature = F.average(variable, axis=1, keepdims=True)

            visualization = self.traverse_computational_graph(variable.creator, averaged_feature)
            visualization = visualization.data
            for i in range(len(visualization)):
                min_val = visualization[i].min()
                max_val = visualization[i].max()
                visualization[i] -= min_val
                visualization[i] *= 1.0 / (max_val - min_val)
        return visualization
