# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from tqdm import *
import random
import numpy as np

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The default behavior for interpolate/upsample with "
                                                                "float scale_factor changed in 1.6.0")
warnings.filterwarnings("ignore", category=UserWarning, message="Default upsampling behavior when mode=bilinear is "
                                                                "changed to align_corners=False")


os.chdir(r'D:\D_project\DCC-net')
def setup_cfg(args):
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
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default=r'D:\D_project\DCC-net\models\R_101_dcni3_5x.yaml',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", default=[r'D:\D_project\DCC-net\static\images\2_test.jpg'], nargs="+")   # severe

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'models/model_upper.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


# 定义一个函数来计算两个点之间的欧几里得距离
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 保证点集长度为10，超出时移除头尾
def ensure_length(points, max_length=10):
    excess_length = len(points) - max_length
    if excess_length > 0:
        # 如果多余的是偶数，从头部和尾部各去除一个点
        if excess_length % 2 == 0:
            temp = excess_length // 2
            points = points[temp:-temp]
        # 如果多余的是奇数，从尾部去除一个点
        else:
            temp = excess_length // 2
            points = points[temp:-temp-1]
    elif len(points) < max_length:
        # 长度不足10，尾部补充0
        points = points + [[0, 0]] * (max_length - len(points))
    return points


# 去除重复和近似的点
def remove_duplicates_and_nearby_points(points, threshold=40):
    unique_points = []
    for point in points:
        # 检查当前点是否与已有的点相近
        if all(euclidean_distance(point, other_point) > threshold for other_point in unique_points):
            unique_points.append(point)
    unique_points = ensure_length(unique_points, max_length=10)

    return unique_points


def crop_with_mask_2(image, predictions):
    target_key = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26]
    # 获取原信息
    instances = predictions['instances']
    pred_boxes_tensor = instances.pred_boxes.tensor.cpu().tolist()
    pred_classes = instances.pred_classes.cpu().tolist()
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
    tempy = image.shape[0]*0.05
    tempx = image.shape[1]*0.08
    y_min, y_max = min_y - tempy, max_y + tempy
    x_min, x_max = min_x - tempx, max_x + tempx
    if y_min<0:
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
            x_point = int(((pred_boxes_tensor[i][0]+pred_boxes_tensor[i][2])/2) - x_min)
            y_point = int(((pred_boxes_tensor[i][1]+pred_boxes_tensor[i][3])/2) - y_min)
            center_points.append([x_point, y_point])
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    center_points = remove_duplicates_and_nearby_points(center_points, threshold=40)
    # print(center_points, len(center_points))
    # # 如果 cropped_image 是只读的，可以复制它
    # if not cropped_image.flags.writeable:
    #     cropped_image = cropped_image.copy()
    # cv2.namedWindow("Image with Point", cv2.WINDOW_NORMAL)
    # for p in center_points:
    #     cv2.circle(cropped_image, p, radius=5, color=(0, 0, 255), thickness=-1)
    # cv2.imshow("Image with Point", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_image, center_points


def crop_with_mask_3(image, predictions):
    # 获取原信息
    instances = predictions['instances']
    masks = np.asarray(instances.pred_masks.cpu())
    masks = masks.astype(int)

    # 创建一个全黑的背景图像（和原图同样大小）
    cropped_image = np.zeros_like(image)
    # 使用掩膜提取原图的对应区域
    for mask in masks:
        mask_region = mask.astype(bool)
        cropped_image[mask_region] = image[mask_region]  # 将原图区域复制到新图像中
    target_key = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26]
    # 获取原信息
    instances = predictions['instances']
    pred_boxes_tensor = instances.pred_boxes.tensor.cpu().tolist()
    pred_classes = instances.pred_classes.cpu().tolist()
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
    tempy = image.shape[0]*0.05
    tempx = image.shape[1]*0.08
    y_min, y_max = min_y - tempy, max_y + tempy
    x_min, x_max = min_x - tempx, max_x + tempx
    if y_min<0:
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
            x_point = int(((pred_boxes_tensor[i][0]+pred_boxes_tensor[i][2])/2) - x_min)
            y_point = int(((pred_boxes_tensor[i][1]+pred_boxes_tensor[i][3])/2) - y_min)
            center_points.append([x_point, y_point])
    cropped_image = cropped_image[int(y_min):int(y_max), int(x_min):int(x_max)]
    center_points = remove_duplicates_and_nearby_points(center_points, threshold=40)
    # print(center_points, len(center_points))
    # # 如果 cropped_image 是只读的，可以复制它
    # if not cropped_image.flags.writeable:
    #     cropped_image = cropped_image.copy()
    # cv2.namedWindow("Image with Point", cv2.WINDOW_NORMAL)
    # for p in center_points:
    #     cv2.circle(cropped_image, p, radius=5, color=(0, 0, 255), thickness=-1)
    # cv2.imshow("Image with Point", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_image, center_points



def get_center(masks):
    center = []
    # 获取每个mask的中心坐标
    for mask in masks:
        y_indices, x_indices = np.nonzero(mask)
        # 计算中心坐标
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)
        center.append([center_x, center_y])
    return center

def get_center_one(mask):
    y_indices, x_indices = np.nonzero(mask)
    center_x = np.mean(x_indices)
    center_y = np.mean(y_indices)
    return center_x, center_y


def func_filter(masks, scores, classes, iou_threshold=0.35):
    filtered_masks, filtered_scores, filtered_classes = zip(
        *[(x, y, z) for x, y, z in zip(masks, scores, classes) if y >= iou_threshold]
    )
    # 转回列表形式
    masks = list(filtered_masks)
    scores = list(filtered_scores)
    classes = list(filtered_classes)
    return masks, scores, classes


def compute_iou(mask1, mask2):
    """计算两个二值掩码之间的交并比（IoU）。"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def mask_nms(masks, scores, classes, iou_threshold=0.5):
    masks, scores, classes = func_filter(masks, scores, classes, iou_threshold=0.35)
    sorted_indices = np.argsort(scores)[::-1]
    masks = [masks[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    classes = [classes[i] for i in sorted_indices]
    keep = [True]*len(masks)
    for i in range(len(masks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks)):
            if not keep[j]:
                continue
            iou = compute_iou(masks[i], masks[j])
            if iou > iou_threshold:
                    keep[j] = False  # 去除置信度较低的mask
    # 过滤保留的mask和对应的置信度
    filtered_masks = [masks[i] for i in range(len(masks)) if keep[i]]
    filtered_scores = [scores[i] for i in range(len(scores)) if keep[i]]
    filtered_classes = [classes[i] for i in range(len(classes)) if keep[i]]
    return filtered_masks, filtered_scores, filtered_classes


# def func_getclass(label_name, mask, middle_line, flag=True):  # True是下颌
#     list1 = [1,2,3,4,5,6,7,8, 17,18,19,20,21]
#     list2 = [9,10,11,12,13,14,15,16, 22,23,24,25,26]
#     c_x, _ = get_center_one(mask)
#     if flag:
#         if c_x > middle_line:
#             if int(label_name) in list1:
#                 if int(label_name) <= 16:
#                     temp = int(label_name) + 30
#                 else:
#                     temp = int(label_name) + 54
#             else:
#                 if int(label_name) <= 16:
#                     temp = int(label_name) - 8 + 30
#                 else:
#                     temp = int(label_name) - 5 + 54
#         else:
#             if int(label_name) in list2:
#                 if int(label_name) <= 16:
#                     temp = int(label_name) + 32
#                 else:
#                     temp = int(label_name) + 59
#             else:
#                 if int(label_name) <= 16:
#                     temp = int(label_name) + 8 + 32
#                 else:
#                     temp = int(label_name) + 5 + 59
#     else:
#         if c_x > middle_line:
#             if int(label_name) in list2:
#                 if int(label_name) <= 16:
#                     temp = int(label_name) + 12
#                 else:
#                     temp = int(label_name) + 39
#             else:
#                 if int(label_name) <= 16:
#                     temp = int(label_name) + 8 + 12
#                 else:
#                     temp = int(label_name) + 5 + 39
#         else:
#             if int(label_name) in list1:
#                 if int(label_name) <= 16:
#                     temp = int(label_name) + 10
#                 else:
#                     temp = int(label_name) + 34
#             else:
#                 if int(label_name) <= 16:
#                     temp = int(label_name) - 8 + 10
#                 else:
#                     temp = int(label_name) - 5 + 34
#     return temp


def find_duplicate_indices(input_list):
    """
    查找列表中重复元素的索引。

    参数:
        input_list (list): 输入列表。

    返回:
        list: 所有重复元素的索引。
    """
    from collections import defaultdict

    # 创建一个字典以存储每个元素的索引
    index_dict = defaultdict(list)
    for i, num in enumerate(input_list):
        index_dict[num].append(i)

    # 提取重复元素的索引
    duplicate_indices = [indices for indices in index_dict.values() if len(indices) > 1]

    # 将索引列表展开为单个列表
    result = [index for sublist in duplicate_indices for index in sublist]

    return result


def check_h_value(out_classes, index_list, masks, flag=True):
    y_value = [get_center_one(masks[idx])[1] for idx in index_list]
    y_max_index = y_value.index(max(y_value))
    y_min_index = y_value.index(min(y_value))
    i_max = index_list[y_max_index]
    i_min = index_list[y_min_index]
    i_min_in_index = index_list.index(i_min)
    i_max_in_index = index_list.index(i_max)
    # cla_max, cla_min = out_classes[i_max], out_classes[i_min]
    down = [38, 75, 48, 85]
    up = [18, 55, 28, 65]
    if flag:
        if out_classes[i_min] in down:
            del index_list[i_min_in_index]
            for count, i in enumerate(index_list, start=1):
                out_classes[i] = out_classes[i] - count
        else:
            del index_list[i_max_in_index]
            for count, i in enumerate(index_list, start=1):
                out_classes[i] = out_classes[i] + count
    else:
        if out_classes[i_max] in up:
            del index_list[i_max_in_index]
            for count, i in enumerate(index_list, start=1):
                out_classes[i] = out_classes[i] - count
        else:
            del index_list[i_min_in_index]
            for count, i in enumerate(index_list, start=1):
                out_classes[i] = out_classes[i] + count
    return out_classes


def func_getclass(classes, masks, middle_line, flag=False):
    out_classes = []
    list1 = [1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21]
    list2 = [9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26]
    for idx, cla in enumerate(classes):
        c_x, _ = get_center_one(masks[idx])
        if flag:
            if c_x > middle_line:
                if int(cla) in list1:
                    if int(cla) <= 16:
                        temp = int(cla) + 30
                    else:
                        temp = int(cla) + 54
                else:
                    if int(cla) <= 16:
                        temp = int(cla) - 8 + 30
                    else:
                        temp = int(cla) - 5 + 54
            else:
                if int(cla) in list2:
                    if int(cla) <= 16:
                        temp = int(cla) + 32
                    else:
                        temp = int(cla) + 59
                else:
                    if int(cla) <= 16:
                        temp = int(cla) + 8 + 32
                    else:
                        temp = int(cla) + 5 + 59
        else:
            if c_x > middle_line:
                if int(cla) in list2:
                    if int(cla) <= 16:
                        temp = int(cla) + 12
                    else:
                        temp = int(cla) + 39
                else:
                    if int(cla) <= 16:
                        temp = int(cla) + 8 + 12
                    else:
                        temp = int(cla) + 5 + 39
            else:
                if int(cla) in list1:
                    if int(cla) <= 16:
                        temp = int(cla) + 10
                    else:
                        temp = int(cla) + 34
                else:
                    if int(cla) <= 16:
                        temp = int(cla) - 8 + 10
                    else:
                        temp = int(cla) - 5 + 34
        out_classes.append(temp)
    index_list = find_duplicate_indices(out_classes)
    out_classes = check_h_value(out_classes, index_list, masks, flag=flag)
    return out_classes



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    path = r'D:\D_project\DCC-net\static\images\2_test.jpg'
    img = read_image(path, format="BGR")
    start_time = time.time()
    predictions, _ = demo.run_on_image(img)
    print(predictions)

    # target = r'D:\D_FILE\data\down\all_image'
    # if args.input:
    #     print(args.input)
    #     if os.path.isdir(args.input[0]):
    #         args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    #     elif len(args.input) == 1:
    #         args.input = glob.glob(os.path.expanduser(args.input[0]))
    #         assert args.input, "The input path(s) was not found"
    #     for path in tqdm.tqdm(args.input, disable=not args.output):
    #         # use PIL, to be consistent with evaluation
    #         name = os.path.basename(path)
    #         # path_t = os.path.join(target, name)
    #         img = read_image(path, format="BGR")
    #         start_time = time.time()
    #         predictions, _ = demo.run_on_image(img)
    #         output_image, center_points = crop_with_mask_2(img, predictions)
    #         output_image, _ = crop_with_mask_3(img, predictions)
    #         logger.info(
    #             "{}: detected {} instances in {:.2f}s".format(
    #                 path, len(predictions["instances"]), time.time() - start_time
    #             )
    #         )
    #
    #         if args.output:
    #             if os.path.isdir(args.output):
    #                 assert os.path.isdir(args.output), args.output
    #                 out_filename = os.path.join(args.output, os.path.basename(path))
    #                 # out_filename = os.path.join(r'D:\project_seg\AdelaiDet\outputs\test3', os.path.basename(path))
    #             else:
    #                 assert len(args.input) == 1, "Please specify a directory with args.output"
    #                 out_filename = args.output
    #             visualized_output.save(out_filename)
    #             cv2.imwrite(out_filename, visualized_output.get_image()[:, :, ::-1])
    #             label_name = os.path.basename(path).split('.')[0] + '.txt'
    #             out_label_path = os.path.join(r'D:\D_FILE\data\down\dataset\train\label\severe', label_name)  # normal
    #             out_image_path = os.path.join(r'D:\D_FILE\data\down\all_seg', name)
    #             print(out_image_path)
    #             cv2.imwrite(out_image_path, output_image)
    #             # 保存label
    #             with open(out_label_path, 'w') as file:
    #                 # 将每个坐标对展开为单独的元素，并用空格连接
    #                 file.write(' '.join(map(str, [coord for pair in center_points for coord in pair])))
    #         else:
    #             cv2.imshow(WINDOW_NAME, output_image)
    #             if cv2.waitKey(0) == 27:
    #                 break  # esc to quit
    #     exit()
