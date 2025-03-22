import os
import random
import argparse
import numpy as np
import warnings
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from torch.utils.data import DataLoader
from model import FasterRCNNVGG16

from data.dataset import RCNNDetectionDataset, RCNNAnnotationTransform
from data import config

warnings.filterwarnings("ignore")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument("--ckpt_path", default="../weights/DOAM/OPIX.pth", type=str, 
                    help="the checkpoint path of the model")

parser.add_argument('--dataset', default="OPIXray", type=str, 
                    choices=["OPIXray", "HiXray", "XAD"], help='Dataset name')
parser.add_argument('--phase', default='test', type=str,
                    help='test phase')

parser.add_argument('--batch_size', default=1, type=int,
                    help='The size of a mini batch')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')


args = parser.parse_args()

fix_seed(0)

torch.set_default_tensor_type('torch.FloatTensor')

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
    
    
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    

def cal_ap(boxes, gts, npos, name, ovthresh=0.5):
    full_boxes = []
    for elm in boxes:
        if len(elm) > 0:
            full_boxes.append(elm)
    if len(full_boxes) == 0:
        return 0.0, 0, 0
    boxes = np.concatenate(full_boxes, 0) # [num_images*num_boxes, 5]
    
    # sort by confidence
    confidence = boxes[:, 4]
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    image_ids = [int(boxes[x, 5]) for x in sorted_ind]
    BB = boxes[sorted_ind, :]
    
    # mark TPs and FPs
    nd = len(image_ids)

    tp = np.zeros(nd)
    fp = np.zeros(nd)

    avg_conf = 0
    tp_num = 0
    
    count = 0

    for d in range(nd):
        gt = np.array(gts[image_ids[d]])
        if len(gt) == 0:
            fp[d] = 1.
            count += 1
            continue
        bb = BB[d, :4].astype(float)
        ovmax = -np.inf
        BBGT = gt[:, :4].astype(float)

        # compute overlaps
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
               (BBGT[:, 2] - BBGT[:, 0]) *
               (BBGT[:, 3] - BBGT[:, 1]) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if gts[image_ids[d]][jmax][5] == 0:
                tp[d] = 1.
                gts[image_ids[d]][jmax][5] = 1
                avg_conf += BB[d, 4]
                tp_num += 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    
    rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec, False)
    return ap, tp_num, count


def cal_map(all_boxes, all_gts, labelmap):
    """
    calculate mAP as PASCAL VOC 2010
    params:
        all_boxes: 
            needs shape as (num_classes, num_images, num_boxes, 6)
            6 means [x1, y1, x2, y2, conf, img_id]
        all_gts:
            needs shape as (num_classes, num_images, num_boxes, 6)
            6 means [x1, y1, x2, y2, label, is_chosen(default to be 0)]
    """
    mAP = 0
    total = 0
    total_tp = 0
    total_fp = 0
    total_missed = 0  # Missed Detection 카운터 추가
    
    print("labelmap:{}".format(labelmap))
    for i, cls in enumerate(labelmap):
        npos = 0
        missed = 0  # 각 클래스별 Missed Detection 카운터
        for elm in all_gts[i]:
            npos += len(elm)
        if npos == 0:
            continue
            
        # Missed Detection 계산
        for img_idx in range(len(all_gts[i])):
            if len(all_gts[i][img_idx]) > 0:  # Ground Truth가 있는 경우
                if len(all_boxes[i][img_idx]) == 0:  # 탐지 결과가 없는 경우
                    missed += len(all_gts[i][img_idx])
                    
        ap, tp, fp = cal_ap(all_boxes[i], all_gts[i], npos, cls)
        print("AP for {}: {:.4f}".format(cls, ap))
        print("True Positives for {}: {}".format(cls, tp))
        print("False Positives for {}: {}".format(cls, fp))
        print("Missed Detections for {}: {}".format(cls, missed))
        
        total_tp += tp
        total_fp += fp
        total_missed += missed
        
        if not np.isnan(ap):
            mAP += ap
            total += 1
            
    print("\n=== Overall Results ===")
    print("mAP: {:.4f}".format(mAP / total))
    print("Total True Positives: {}".format(total_tp))
    print("Total False Positives: {}".format(total_fp))
    print("Total Missed Detections: {}".format(total_missed))
    print("Detection Rate: {:.2f}%".format((total_tp / (total_tp + total_missed)) * 100 if (total_tp + total_missed) > 0 else 0))


def save_detection_results(image, boxes, labels, scores, labelmap, save_path, gt_boxes=None, gt_labels=None, conf_thresh=0.5):
    """
    객체 탐지 결과를 이미지에 그려서 저장하는 함수
    Args:
        image: 원본 이미지 (numpy array)
        boxes: 바운딩 박스 좌표 (numpy array)
        labels: 클래스 레이블 (numpy array)
        scores: 신뢰도 점수 (numpy array)
        labelmap: 클래스 이름 리스트
        save_path: 저장할 경로
        gt_boxes: Ground Truth 바운딩 박스 좌표 (numpy array)
        gt_labels: Ground Truth 클래스 레이블 (numpy array)
        conf_thresh: 신뢰도 임계값
    """
    # matplotlib figure 생성
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    
    # 색상 맵 생성
    colors = plt.cm.hsv(np.linspace(0, 1, len(labelmap)))
    
    # Ground Truth 표시 (점선)
    if gt_boxes is not None and gt_labels is not None:
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            color = colors[label]
            
            # Ground Truth 바운딩 박스 그리기 (점선)
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color=color, linewidth=2, linestyle='--')
            plt.gca().add_patch(rect)
            
            # Ground Truth 레이블 표시
            label_text = f'GT: {labelmap[label]}'
            plt.text(x1, y1-5, label_text, 
                    color=color, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # 검출 결과 표시 (실선)
    detected_gt = set()  # 이미 탐지된 Ground Truth 인덱스 추적
    
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_thresh:
            continue
            
        x1, y1, x2, y2 = box
        color = colors[label]
        
        # Ground Truth와의 IoU 계산
        if gt_boxes is not None and gt_labels is not None:
            max_iou = 0
            max_gt_idx = -1
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx in detected_gt:
                    continue
                    
                # IoU 계산
                ixmin = max(x1, gt_box[0])
                iymin = max(y1, gt_box[1])
                ixmax = min(x2, gt_box[2])
                iymax = min(y2, gt_box[3])
                iw = max(ixmax - ixmin, 0)
                ih = max(iymax - iymin, 0)
                inters = iw * ih
                uni = ((x2 - x1) * (y2 - y1) + 
                      (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) - inters)
                iou = inters / uni if uni > 0 else 0
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # IoU가 0.5 이상이면 올바른 탐지로 간주
            if max_iou >= 0.5:
                detected_gt.add(max_gt_idx)
                status = "Correct"
            else:
                status = "Wrong"
        else:
            status = "Detected"
        
        # 검출 결과 바운딩 박스 그리기 (실선)
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, color=color, linewidth=2)
        plt.gca().add_patch(rect)
        
        # 검출 결과 레이블과 신뢰도 점수 표시
        label_text = f'{status} {labelmap[label]}: {score:.2f}'
        plt.text(x1, y1-5, label_text, 
                color=color, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # 탐지되지 않은 Ground Truth 표시
    if gt_boxes is not None and gt_labels is not None:
        for gt_idx, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx not in detected_gt:
                x1, y1, x2, y2 = box
                color = colors[label]
                
                # 탐지 실패한 객체 표시 (빨간색 점선)
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2, linestyle='--')
                plt.gca().add_patch(rect)
                
                # 탐지 실패 레이블 표시
                label_text = f'Missed: {labelmap[label]}'
                plt.text(x1, y1-5, label_text, 
                        color='red', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.7))
    
    # 이미지 저장
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def test_net(net, cuda, dataset, labelmap, im_size=300):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap))]
    
    all_gts = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap))]

    # 결과 저장 디렉토리 생성
    save_dir = "detection_results"
    os.makedirs(save_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    nob = 0
    avg_conf = 0
    for i, (images, bboxes, labels, scale, img_id) in enumerate(tqdm(loader)):
        x = images.type(torch.cuda.FloatTensor)
        
        # Ground Truth 저장
        for j in range(images.shape[0]):
            for k in range(bboxes[j].shape[0]):
                bbox = bboxes[j][k].numpy().tolist()
                label = labels[j][k].item()
                t = i * args.batch_size + j
                all_gts[label][t].append([bbox[0], bbox[1], bbox[2], bbox[3], label, 0])

        with torch.no_grad():
            boxes_, labels_, scores_ = net.predict(x, sizes=[images[zz].shape[1:] for zz in range(images.shape[0])])
        
        # 각 이미지에 대한 검출 결과 저장
        for k in range(len(labels_)):
            for m in range(len(labels_[k])):
                boxes = boxes_[k][m]
                labels = labels_[k][m]
                scores = scores_[k][m]

                img_ids = i * args.batch_size + k
                cls_dets = torch.tensor([boxes[0], boxes[1], boxes[2], boxes[3], scores, img_ids])
                all_boxes[labels][i*args.batch_size+k].append(cls_dets.cpu().numpy())
                
                # 원본 이미지 가져오기
                img = images[k].cpu().numpy().transpose(1, 2, 0)
                
                # 이미지 정규화 해제
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = ((img * std + mean) * 255).clip(0, 255).astype(np.uint8)
                
                # Ground Truth 정보 가져오기
                gt_boxes = bboxes[k].numpy()
                # labels가 스칼라인 경우 처리
                if isinstance(labels, torch.Tensor):
                    gt_labels = labels.numpy()
                else:
                    gt_labels = np.array([labels])
                
                # 검출 결과 저장
                save_path = os.path.join(save_dir, f"detection_{img_ids}.jpg")
                
                # 단일 값들을 리스트로 변환
                boxes = np.array([boxes])
                labels = np.array([labels])
                scores = np.array([scores])
                
                save_detection_results(img, boxes, labels, scores, labelmap, save_path, gt_boxes, gt_labels)
                
                nob += 1
                avg_conf += scores
    
    confs = []
    for elm_cls in all_boxes:
        for elm in elm_cls:
            for e in elm:
                confs.append(e[4])
    cal_map(all_boxes, all_gts, labelmap)


if __name__ == '__main__':
    print(args)
    if args.dataset == "OPIXray":
        data_info = config.OPIXray_test
    elif args.dataset == "HiXray":
        data_info = config.HiXray_test
    elif args.dataset == "XAD":
        data_info = config.XAD_test

    num_classes = len(data_info["model_classes"]) + 1

    net = FasterRCNNVGG16(config.FasterRCNN, num_classes - 1)
    
    # 모델 로드 수정 (DataParallel 저장된 모델을 일반 모델로 변환)
    state_dict = torch.load(args.ckpt_path, map_location=torch.device('cuda' if args.cuda else 'cpu'))
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # "module." 제거
    net.load_state_dict(new_state_dict)

    ''' 원본코드
    net.load_state_dict(torch.load(args.ckpt_path))
    net.eval()
    '''
    
    dataset = RCNNDetectionDataset(root=data_info["dataset_root"], 
                            model_classes=data_info["model_classes"],
                            image_sets=data_info["imagesetfile"], 
                            target_transform=RCNNAnnotationTransform(data_info["model_classes"]), 
                            phase=args.phase)

    if args.cuda:
        net = net.cuda()
        
    test_net(net, args.cuda, dataset, data_info["model_classes"], 300)
    print(args.ckpt_path, args.phase)