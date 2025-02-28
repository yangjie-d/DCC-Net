import torch
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from tqdm import *
import random
import numpy as np
import torch.nn as nn

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from AdelaiDet.model_pred.predictor import VisualizationDemo
from AdelaiDet.adet.config import get_cfg


class Diagnose_model:
    # 初始化方法
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg1 = self.setup_cfg(self.get_parser(True).parse_args(), )    # True是上
        self.model_up = VisualizationDemo(cfg1)
        cfg2 = self.setup_cfg(self.get_parser(False).parse_args(), )
        self.model_down = VisualizationDemo(cfg2)
        self.model_cla_down = torch.load('AdelaiDet/weights/down_method2.pth').to(self.device).eval()
        self.model_cla_up = torch.load('AdelaiDet/weights/up_method2.pth').to(self.device).eval()

    def run_model(self, img, flag=True):  # True是上
        img = self.convert_PIL_to_numpy(img, format='BGR')
        if flag:
            predictions, _ = self.model_up.run_on_image(img)
            size = (320, 224)
        else:
            predictions, _ = self.model_down.run_on_image(img)
            size = (320, 160)
        cropped_image, center_points, box, masks = self.get_center_and_roi(img, predictions)
        points_list = []
        for points in center_points:
            points_list.append(points[0])
            points_list.append(points[1])
        image_tensor, coordinate_tensor = self.create_input_data(cropped_image, points_list, size=size)
        if flag:
            output = self.model_cla_up(image_tensor.unsqueeze(0).to(self.device).float(),
                                         coordinate_tensor.unsqueeze(0).to(self.device).float())
        else:
            output = self.model_cla_down(image_tensor.unsqueeze(0).to(self.device).float(),
                                       coordinate_tensor.unsqueeze(0).to(self.device).float())
        pred_score = nn.functional.softmax(output, dim=1)
        pred_classes = pred_score.argmax(dim=1)

        box = [box[2], box[0], box[3], box[1]]

        return pred_score, pred_classes, center_points, box, masks

    def create_input_data(self, img, keypoints, size=(320, 160)):
        img_array = np.array(img)
        # 调整尺寸,调整坐标
        resized_image, new_label = self.resize_image_and_keypoints(img_array, keypoints, size[0], size[1])

        # 坐标归一化
        new_label = self.normalization(new_label, size=size)
        # 图像归一化
        new_image = resized_image / 255.0
        new_image = np.transpose(new_image, (2, 0, 1))
        new_image = torch.tensor(new_image, dtype=torch.float32)
        new_label = torch.tensor(new_label, dtype=torch.float32)  # 转换坐标为 Tensor
        return new_image, new_label

    def normalization(self, keypoints, size=(256, 128)):
        w, h = size
        new_keypoints = []
        for i in range(0, len(keypoints), 2):
            x, y = keypoints[i], keypoints[i + 1]
            if x != 0 and y != 0:
                new_x = round(x / w, 8)
                new_y = round(y / h, 8)
                new_keypoints.extend([new_x, new_y])
            else:
                new_keypoints.extend([x, y])  # 保持0不变
        return new_keypoints

    def convert_PIL_to_numpy(self, image, format):
        """
        Convert PIL image to numpy array of target format.

        Args:
            image (PIL.Image): a PIL image
            format (str): the format of output image

        Returns:
            (np.ndarray): also see `read_image`
        """
        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format in ["BGR", "YUV-BT.601"]:
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]

        return image

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
        cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.MODEL.BASIS_MODULE.NUM_CLASSES = 27
        cfg.freeze()
        return cfg

    def get_parser(self, flag):
        parser = argparse.ArgumentParser(description="Detectron2 Demo")
        parser.add_argument(
            "--config-file",
            default=r'AdelaiDet/configs/BlendMask/R_101_dcni3_5x.yaml',
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.3,
            help="Minimum score for instance predictions to be shown",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=['MODEL.WEIGHTS', 'AdelaiDet/weights/model_upper.pth'] if flag else ['MODEL.WEIGHTS', 'AdelaiDet/weights/model_lower.pth'],
            nargs=argparse.REMAINDER,
        )
        return parser

    # 去除重复和近似的点
    def remove_duplicates_and_nearby_points(self, points, threshold=40):
        unique_points = []
        for point in points:
            # 检查当前点是否与已有的点相近
            if all(self.euclidean_distance(point, other_point) > threshold for other_point in unique_points):
                unique_points.append(point)
        unique_points = self.ensure_length(unique_points, max_length=10)

        return unique_points

    # 定义一个函数来计算两个点之间的欧几里得距离
    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # 保证点集长度为10，超出时移除头尾
    def ensure_length(self, points, max_length=10):
        excess_length = len(points) - max_length
        if excess_length > 0:
            # 如果多余的是偶数，从头部和尾部各去除一个点
            if excess_length % 2 == 0:
                temp = excess_length // 2
                points = points[temp:-temp]
            # 如果多余的是奇数，从尾部去除一个点
            else:
                temp = excess_length // 2
                points = points[temp:-temp - 1]
        elif len(points) < max_length:
            # 长度不足10，尾部补充0
            points = points + [[0, 0]] * (max_length - len(points))
        return points

    def compute_iou(self, mask1, mask2):
        """计算两个二值掩码之间的交并比（IoU）。"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    def mask_nms(self, masks, scores, classes, pred_boxes, iou_threshold=0.5):
        sorted_indices = np.argsort(scores)[::-1]
        masks = [masks[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        classes = [classes[i] for i in sorted_indices]
        pred_boxes = [pred_boxes[i] for i in sorted_indices]
        keep = [True] * len(masks)
        for i in range(len(masks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(masks)):
                if not keep[j]:
                    continue
                iou = self.compute_iou(masks[i], masks[j])
                if iou > iou_threshold:
                    keep[j] = False  # 去除置信度较低的mask
        # 过滤保留的mask和对应的置信度
        filtered_masks = [masks[i] for i in range(len(masks)) if keep[i]]
        filtered_scores = [scores[i] for i in range(len(scores)) if keep[i]]
        filtered_classes = [classes[i] for i in range(len(classes)) if keep[i]]
        filtered_boxes = [pred_boxes[i] for i in range(len(pred_boxes)) if keep[i]]
        return filtered_masks, filtered_boxes, filtered_classes, filtered_scores

    def get_center_and_roi(self, image, predictions):
        # 获取原信息
        instances = predictions['instances']
        masks = np.asarray(instances.pred_masks.cpu()).astype(int)
        scores = instances.scores.cpu().tolist()
        pred_classes = instances.pred_classes.cpu().tolist()
        pred_boxes = instances.pred_boxes.tensor.cpu().tolist()
        selected_masks, selected_boxes, selected_classes, selected_scores = self.mask_nms(masks, scores, pred_classes,
                                                                                     pred_boxes,
                                                                                     iou_threshold=0.5)

        target_key = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21,
                      22, 23, 24, 25, 26]
        pred_boxes_tensor = selected_boxes
        pred_classes = selected_classes
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        center_points = []
        # 对每个目标类别处理
        for i, t in enumerate(pred_boxes_tensor):
            if pred_classes[i] in target_key:
                if pred_boxes_tensor[i][0] < min_x:
                    min_x = pred_boxes_tensor[i][0]
                if pred_boxes_tensor[i][1] < min_y:
                    min_y = pred_boxes_tensor[i][1]
                if pred_boxes_tensor[i][2] > max_x:
                    max_x = pred_boxes_tensor[i][2]
                if pred_boxes_tensor[i][3] > max_y:
                    max_y = pred_boxes_tensor[i][3]
        tempy = image.shape[0] * 0.05
        tempx = image.shape[1] * 0.08
        y_min, y_max = min_y - tempy, max_y + tempy
        x_min, x_max = min_x - tempx, max_x + tempx
        if y_min < 0:
            y_min = 0
        if x_min < 0:
            x_min = 0
        if y_max > image.shape[0]:
            y_max = image.shape[0]
        if x_max > image.shape[1]:
            x_max = image.shape[1]
        # 处理坐标
        for i, t in enumerate(pred_boxes_tensor):
            if pred_classes[i] in target_key:
                x_point = int(((pred_boxes_tensor[i][0]+pred_boxes_tensor[i][2])/2))
                y_point = int(((pred_boxes_tensor[i][1]+pred_boxes_tensor[i][3])/2))
                # x_point = int(((pred_boxes_tensor[i][0] + pred_boxes_tensor[i][2]) / 2) - x_min)
                # y_point = int(((pred_boxes_tensor[i][1] + pred_boxes_tensor[i][3]) / 2) - y_min)
                center_points.append([x_point, y_point])
        # box = [int(x_min), int(y_min), int(x_max), int(y_max)]
        box = [int(y_min), int(y_max), int(x_min), int(x_max)]
        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        center_points = self.remove_duplicates_and_nearby_points(center_points, threshold=40)
        # return out_image, center_points, box
        return cropped_image, center_points, box, masks

    def draw_box_and_points(self, img, box, center_points, predictions, box_color=(144, 238, 144), point_color=(0, 255, 0)):
        """
        在图像上绘制矩形和点，线宽和点半径根据图像分辨率自适应。

        参数:
            img (numpy.ndarray): 输入的图像。
            box (list): 矩形框的坐标 [y_min, y_max, x_min, x_max]。
            center_points (list): 点的坐标列表，每个点是 [x, y]。
            box_color (tuple): 矩形框的颜色 (B, G, R)，默认浅绿色。
            point_color (tuple): 点的颜色 (B, G, R)，默认红色。

        返回:
            numpy.ndarray: 绘制后的图像。
        """
        img = img.copy()  # 创建一个新的可写副本

        # 获取图像分辨率
        height, width = img.shape[:2]

        # 计算自适应线宽和点半径
        thickness = max(1, int((width + height) / 400))  # 确保最小线宽为 1
        point_radius = max(2, int((width + height) / 300))  # 确保最小点半径为 2
        mask_image = self.draw_mask(img, predictions, alpha=0.3)
        img4 = mask_image.copy()

        # 绘制矩形框 method1
        cv2.rectangle(
            img,
            (int(box[2]), int(box[0])),  # 左上角坐标 (x_min, y_min)
            (int(box[3]), int(box[1])),  # 右下角坐标 (x_max, y_max)
            color=box_color,
            thickness=thickness
        )
        img2 = img.copy()

        # 绘制点  method2
        for point in center_points:
            cv2.circle(
                img2,
                (int(point[0]), int(point[1])),  # 点的坐标 (x, y)
                radius=point_radius,
                color=point_color,
                thickness=-1  # 填充点
            )

        # mask + roi
        cv2.rectangle(
            mask_image,
            (int(box[2]), int(box[0])),  # 左上角坐标 (x_min, y_min)
            (int(box[3]), int(box[1])),  # 右下角坐标 (x_max, y_max)
            color=box_color,
            thickness=thickness
        )

        img3 = mask_image.copy()
        for point in center_points:
            cv2.circle(
                img3,
                (int(point[0]), int(point[1])),  # 点的坐标 (x, y)
                radius=point_radius,
                color=point_color,
                thickness=-1  # 填充点
            )
        # 调用函数示例
        self.show_four_images(img, img2, img3, img4)

        return img

    def show_four_images(self, img1, img2, img3, img4):
        # if img1 is None or img2 is None or img3 is None or img4 is None:
        #     print("Error: One or more image paths are invalid or images cannot be loaded.")
        #     return
        #
        # 拼接图片
        top_row = np.hstack((img1, img2))
        bottom_row = np.hstack((img3, img4))
        combined_image = np.vstack((top_row, bottom_row))

        # 显示拼接后的图片
        cv2.namedWindow('s', cv2.WINDOW_NORMAL)
        cv2.imshow('s', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_mask(self, image, predictions, alpha=0.3):
        # 创建图像的副本
        output = image.copy()
        # 获取实例信息
        instances = predictions['instances']
        masks = np.asarray(instances.pred_masks.cpu())
        masks = masks.astype(int)
        # 遍历每个 mask 并绘制到图像上
        for idx in range(len(masks)):
            # 生成随机颜色
            color = [random.randint(0, 255) for _ in range(3)]
            # 当前 mask
            mask = masks[idx].astype(np.uint8)
            # 创建彩色 mask
            color_mask = np.zeros_like(image, dtype=np.uint8)
            for i in range(3):
                color_mask[:, :, i] = mask * color[i]
            # 创建 alpha mask (透明度叠加区域)
            alpha_mask = mask[:, :, None] * alpha
            # 将彩色 mask 与原图叠加（值为 0 的区域完全透明）
            output = (1 - alpha_mask) * output + alpha_mask * color_mask
            output = output.astype(np.uint8)

        # # 显示图像
        # cv2.namedWindow("Image with Mask", cv2.WINDOW_NORMAL)
        # cv2.imshow("Image with Mask", output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return output

    def resize_image_and_keypoints(self, image, keypoints, new_width, new_height):
        # 获取原图尺寸
        old_height, old_width = image.shape[:2]

        # 计算宽度和高度的缩放比例
        width_scale = new_width / float(old_width)
        height_scale = new_height / float(old_height)

        # 调整关键点坐标
        new_keypoints = []
        for i in range(0, len(keypoints), 2):
            x, y = keypoints[i], keypoints[i + 1]
            if x != 0 and y != 0:
                new_x = round(x * width_scale)
                new_y = round(y * height_scale)
                new_keypoints.extend([new_x, new_y])
            else:
                new_keypoints.extend([x, y])  # 保持0不变

        # 调整图片尺寸
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image, new_keypoints


