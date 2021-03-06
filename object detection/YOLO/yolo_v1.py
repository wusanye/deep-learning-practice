"""
YOLO v1 by tensorflow,
Here is the inference assuming the train is done
by w.k. 2018.07.15
"""

import numpy as np
import tensorflow as tf
import cv2


def leak_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


class Yolo(object):
    def __init__(self, weight_file):
        self.verbose = True
        # detection params
        self.S = 7
        self.B = 2
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]
        self.C = len(self.classes)  # num of classes

        # offset for box center (w.r.t top left point of each cell)
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)] * self.S * self.B), [self.B, self.S, self.S]), [1, 2, 0])  # shape (7, 7, 2)
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])  # shape (7, 7, 2)

        self.threshold = 0.2  # confident scores threshold
        self.iou_threshold = 0.5

        self.sess = tf.Session()
        self._build_net()
        self._load_weights(weight_file)

    def _build_net(self):
        """ build the net"""
        if self.verbose:
            print("Start to bulid the network ...")
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])
        net = self._conv_layer(self.images, 1, 64, 7, 2)
        net = self._maxpool_layer(net, 1, 2, 2)
        net = self._conv_layer(net, 2, 192, 3, 1)
        net = self._maxpool_layer(net, 2, 2, 2)
        net = self._conv_layer(net, 3, 128, 1, 1)
        net = self._conv_layer(net, 4, 256, 3, 1)
        net = self._conv_layer(net, 5, 256, 1, 1)
        net = self._conv_layer(net, 6, 512, 3, 1)
        net = self._maxpool_layer(net, 6, 2, 2)
        net = self._conv_layer(net, 7, 256, 1, 1)
        net = self._conv_layer(net, 8, 512, 3, 1)
        net = self._conv_layer(net, 9, 256, 1, 1)
        net = self._conv_layer(net, 10, 512, 3, 1)
        net = self._conv_layer(net, 11, 256, 1, 1)
        net = self._conv_layer(net, 12, 512, 3, 1)
        net = self._conv_layer(net, 13, 256, 1, 1)
        net = self._conv_layer(net, 14, 512, 3, 1)
        net = self._conv_layer(net, 15, 512, 1, 1)
        net = self._conv_layer(net, 16, 1024, 3, 1)
        net = self._maxpool_layer(net, 16, 2, 2)
        net = self._conv_layer(net, 17, 512, 1, 1)
        net = self._conv_layer(net, 18, 1024, 3, 1)
        net = self._conv_layer(net, 19, 512, 1, 1)
        net = self._conv_layer(net, 20, 1024, 3, 1)
        net = self._conv_layer(net, 21, 1024, 3, 1)
        net = self._conv_layer(net, 22, 1024, 3, 2)
        net = self._conv_layer(net, 23, 1024, 3, 1)
        net = self._conv_layer(net, 24, 1024, 3, 1)
        net = self._flatten(net)
        net = self._fc_layer(net, 25, 512, activation=leak_relu)
        net = self._fc_layer(net, 26, 4096, activation=leak_relu)
        net = self._fc_layer(net, 27, self.S*self.S*(self.C+5*self.B))
        self.predicts = net

    def _conv_layer(self, x, id, filters_num, filter_size, stride):
        """convolution layer"""
        in_channels = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, filters_num], stddev=0.1))
        bias = tf.Variable(tf.zeros([filters_num, ]))

        # padding, note: not using padding='SAME'
        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)  # add pad_size cols or rows for up down left right direction

        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding='VALID')  # strides (batch, height, width, channel)
        output = leak_relu(tf.nn.bias_add(conv, bias))

        if self.verbose:
            print("Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s"
                  % (id, filters_num, filter_size, stride, str(output.get_shape())))

        return output

    def _fc_layer(self, x, id, num_out, activation=None):
        """fully connected layer"""
        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_out, ]))
        output = tf.nn.xw_plus_b(x, weight, bias)
        if activation:
            output = activation(output)
        if self.verbose:
            print("Layer %d: type=Fc, num_out=%d, output_shape=%s"
                  % (id, num_out, str(output.get_shape())))

        return output

    def _maxpool_layer(self, x, id, pool_size, stride):
        """max pooling layer"""
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding='SAME')

        if self.verbose:
            print("Layer %d: type=MaxPool, pool_size=%d, stride=%d, output_shape=%s"
                  % (id, pool_size, stride, str(output.get_shape())))

        return output

    def _flatten(self, x):
        """flatten layer"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channel first mode
        nums = np.prod(x.get_shape().as_list()[1:])  # C*W*H
        return tf.reshape(tran_x, [-1, nums])  # samples * (C*W*H)

    def _load_weights(self, weights_file):
        """load weights from file"""
        if self.verbose:
            print("Start to load weights from file:%s" % (weights_file))
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_file)

    def _interpret_predicts(self, predicts, img_h, img_w):
        """interpret the predicts and get the detection boxes"""
        idx1 = self.S*self.S*self.C
        idx2 = idx1 + self.S*self.S*self.B

        # class predictions
        class_probs = np.reshape(predicts[:idx1], [self.S, self.S, self.C])
        # box confidence
        confs = np.reshape(predicts[idx1:idx2], [self.S, self.S, self.B])
        # boxes -> (x, y, w, h)
        boxes = np.reshape(predicts[idx2:], [self.S, self.S, self.B, 4])

        # convert the x, y to the coordinate relative to the reference point of the image (top left)
        boxes[:, :, :, 0] += self.x_offset  # each box's coordinate is offset relative to the grid cell's top left point
        boxes[:, :, :, 1] += self.y_offset  # each box's coordinate is offset relative to the grid cell's top left point
        boxes[:, :, :, :2] /= self.S        # relative to grid's width and height

        # predictions of w, h are square rooted
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        # scale to original size
        boxes[:, :, :, 0] *= img_w
        boxes[:, :, :, 1] *= img_h
        boxes[:, :, :, 2] *= img_w
        boxes[:, :, :, 3] *= img_h

        # class-specific confidence scores [S, S, B, C]
        scores = np.expand_dims(confs, -1) * np.expand_dims(class_probs, 2)

        scores = np.reshape(scores, [-1, self.C])  # [S*S*B, C]
        boxes = np.reshape(boxes, [-1, 4])  # [S*S*B, 4]

        # filter the boxes with scores less than threshold
        scores[scores < self.threshold] = 0.0

        # non-max suppression
        self._non_max_suppression(scores, boxes)

        # predicted boxes
        predicted_boxes = []  # (class, x, y, w, h, scores)
        max_idxs = np.argmax(scores, axis=1)
        for i in range(len(scores)):
            max_idx = max_idxs[i]
            if scores[i, max_idx] > 0.0:
                predicted_boxes.append((self.classes[max_idx], boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i, max_idx]))

        return predicted_boxes

    def _non_max_suppression(self, scores, boxes):
        """non-max suppression"""
        for c in range(self.C):
            sorted_idxs = np.argsort(scores[:, c])  # sort in ascending order
            last = len(sorted_idxs) - 1  # idx corresponding to the max element
            while last > 0:
                if scores[sorted_idxs[last], c] < 1e-6:
                    break
                for i in range(last):
                    if scores[sorted_idxs[i], c] < 1e-6:
                        continue
                    if self._iou(boxes[sorted_idxs[i]], boxes[sorted_idxs[last]]) > self.iou_threshold:
                        scores[sorted_idxs[i], c] = 0.0
                last -= 1

    def _iou(self, box1, box2):
        """compute the iou between two boxes"""
        intersection_w = np.minimum(box1[0] + .5*box1[2], box2[0] + .5*box2[2]) - \
                         np.maximum(box1[0] - .5*box1[2], box2[0] - .5*box2[2])
        intersection_h = np.minimum(box1[1] + .5*box1[3], box2[1] + .5*box2[3]) - \
                         np.maximum(box1[1] - .5*box1[3], box2[1] - .5*box2[3])
        if intersection_h < 0 or intersection_w < 0:
            intersection = 0
        else:
            intersection = intersection_w * intersection_h

        union = box1[2]*box1[3] + box2[2]*box2[3] - intersection

        iou_ratio = intersection / union

        return iou_ratio

    def detect_from_file(self, image_file, imshow=True, detected_boxes_file='boxes.txt',
                        detected_image_file='detected_image.jpg'):
        """detection given a image file"""
        image = cv2.imread(image_file)
        img_h, img_w, _ = image.shape
        predicts = self.detect_from_image(image)
        predicted_boxes = self._interpret_predicts(predicts, img_h, img_w)

        self.show_results(image, predicted_boxes, imshow, detected_boxes_file, detected_image_file)

    def detect_from_image(self, image):
        """detection given a cv image"""
        img_resized = cv2.resize(image, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32)
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        predicts = self.sess.run(self.predicts, feed_dict={self.images: _images})[0]  # [batch, S*S*(B*5+C)]

        return predicts

    def show_results(self, image, predicted_boxes, imshow=True, detected_boxes_file=None, detected_image_file=None):
        """show the detected boxes"""
        img_cp = image.copy()
        if detected_boxes_file:
            f = open(detected_boxes_file, 'w')

        # draw bounding boxes
        for i in range(len(predicted_boxes)):
            x, y = int(predicted_boxes[i][1]), int(predicted_boxes[i][2])
            w, h = int(predicted_boxes[i][3]) // 2, int(predicted_boxes[i][4]) // 2
            if self.verbose:
                print("class: %s, [x, y, w, h]=[%d, %d, %d, %d], confidence=%f" % (predicted_boxes[i][0],
                        x, y, w, h, predicted_boxes[i][-1]))

                cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w, y + h + 20), (x + w, y + h), (125, 125, 125), -1)
                cv2.putText(img_cp, predicted_boxes[i][0] + ' : %.2f' % predicted_boxes[i][5], (x - w + 5, y + h + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if detected_boxes_file:
                f.write(predicted_boxes[i][0] + ',' + str(x) + ',' + str(y) + ',' +
                        str(w) + ',' + str(h) + ',' + str(predicted_boxes[i][5]) + '\n')

        if imshow:
            cv2.imshow('YoOLO_v1_detection', img_cp)
            cv2.waitKey(1)
        if detected_image_file:
            cv2.imwrite(detected_image_file, img_cp)
        if detected_boxes_file:
            f.close()


if __name__ == "__main__":
    yolo_net = Yolo("./weights/yolo_v1_weights.ckpt")
    yolo_net.detect_from_file("./test/nba.jpg")
    input()




