"""
Xray Adversarial Attack
"""
import os
import sys
import cv2
import math
import time
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from model import FasterRCNNVGG16, FasterRCNNTrainer

from data.dataset import RCNNDetectionDataset, rcnn_detection_collate_attack, RCNNAnnotationTransform
from data import config
from utils import stick, renderer, rl_utils

warnings.filterwarnings("ignore")
torch.set_default_tensor_type('torch.FloatTensor')


parser = argparse.ArgumentParser(description="X-ray adversarial attack.")
# for model
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for the experiments')
parser.add_argument("--ckpt_path", default="./ckpt/OPIX.pth", type=str, 
                    help="the checkpoint path of the model")
# for data
parser.add_argument('--dataset', default="OPIXray", type=str, 
                    choices=["OPIXray", "HiXray"], help='Dataset name')
parser.add_argument("--phase", default="test", type=str, 
                    help="the phase of the X-ray image dataset")
parser.add_argument("--batch_size", default=10, type=int, 
                    help="the batch size of the data loader")
parser.add_argument("--num_workers", default=4, type=int, 
                    help="the number of workers of the data loader")
# for patch
parser.add_argument("--obj_path", default="objs/simple_door_key2.obj", type=str, #초기 3D 적대적 객체 obj 파일의 위치 지정
                    help="the path of adversarial 3d object file")
parser.add_argument("--patch_size", default=35, type=int, #패치의 사이즈
                    help="the size of X-ray patch")
parser.add_argument("--patch_count", default=1, type=int, #적대적 패치의 숫자
                    help="the number of X-ray patch")
parser.add_argument("--patch_place", default="reinforce", type=str, choices=['none', 'fix', 'fix_patch', 'reinforce'],
                    help="the place where the X-ray patch located") #적대적 객체의 위치 지정
parser.add_argument("--patch_material", default="iron", type=str, choices=["iron", "plastic", "aluminum", "iron_fix"],
                    help="the material of patch, which decides the color of patch")  #적대적 객체의 재질 지정   
# for attack
parser.add_argument("--targeted", default=False, action="store_true",
                    help="whether to use targeted (background) attack") #타겟 어택 여부 지정
parser.add_argument("--lr", default=0.01, type=float,  #학습률
                    help="the learning rate of attack")
parser.add_argument("--beta", default=0.01, type=float, 
                    help="the perceptual loss rate of attack") #지각적 손실 (즉, 왜곡에 대한 부분)
parser.add_argument("--num_iters", default=24, type=int, #반복수 설정
                    help="the number of iterations of attack")
parser.add_argument("--save_path", default="./results", type=str,
                    help="the save path of adversarial examples")

timer = time.time()
def stime(content):
    global timer
    torch.cuda.synchronize()
    print(content, time.time() - timer)
    timer = time.time()

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

args = parser.parse_args()
args.save_path = os.path.join(args.save_path, f"{args.dataset}/{args.patch_material}/{args.patch_place}", "FasterRCNN")

fix_seed(args.seed)
print(args)

#---------------------------- new function ----------------------------------
#전체 지각손실함수 부분
def shape_loss(
    vertices_ori, 
    vertices_adv,
    z_weight=1.0,
    mask=None
):
    """
    원본 열쇠 메쉬(vertices_ori)와 업데이트된 메쉬(vertices_adv) 간 L2 거리를 계산하되,
    z축 변화에는 z_weight 가중치를 곱해 두께 변화를 억제하는 간단한 함수.
    """
    # (1) 좌표 차이 계산
    diff = vertices_adv - vertices_ori
    dx = diff[:, 0]
    dy = diff[:, 1]
    dz = diff[:, 2]

    # (2) z축에 가중치 부여
    dist_all = torch.sqrt(dx**2 + dz**2 + (z_weight * dy)**2 + 1e-8)

    # (3) 마스크가 있다면, 그 부분만 평균
    #     예: groove_mask, blade_mask 등
    if mask is not None:
        dist_all = dist_all[mask]
        if dist_all.numel() == 0:
            return torch.tensor(0.0, device=vertices_adv.device)
    
    # (4) 전체 평균
    return dist_all.mean()

def load_groove_coords(txt_path, mean=0.5, std=1/2.4):
    """
    txt 파일에서 'v x y z' 형태의 좌표를 불러오고,
    정규화를 적용하여 (M, 3) shape의 텐서로 반환.
    """
    coords = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if tokens[0] == 'v':
                x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
                coords.append([x, y, z])
    groove_coords = torch.tensor(coords, dtype=torch.float32)
    
    #축 변경
    temp_y = groove_coords[:, 1].clone()
    groove_coords[:, 1] = groove_coords[:, 2]  # Y <- Z
    groove_coords[:, 2] = -temp_y   
    # 정규화 적용: OBJ에서 사용한 것과 동일하게
    groove_coords = (groove_coords * std) + mean
    
    return groove_coords


def create_groove_mask(vertices, groove_coords):
    """
    vertices: (N, 3) - OBJ에서 불러온 전체 정점 (GPU에 있음)
    groove_coords: (M, 3) - txt 파일에서 불러온, 정규화된 홈 정점 좌표 (CPU 또는 GPU)
    
    반환: groove_mask: (N,) bool 텐서
          각 정점이 홈 영역이면 True.
          오직 완전히 동일한 좌표인 경우에만 True로 처리.
    """
    N = vertices.shape[0]
    groove_mask = torch.zeros(N, dtype=torch.bool, device=vertices.device)
    
    # groove_coords를 vertices와 동일한 디바이스로 이동
    groove_coords = groove_coords.to(vertices.device)
    
    # 각 groove 좌표와 정확히 동일한 정점을 찾아 True로 설정
    for gc in groove_coords:
        # 각 정점과 gc가 완전히 같은지 비교 (모든 좌표값이 동일해야 함)
        equal_mask = (vertices == gc).all(dim=1)  # (N,) bool 텐서
        groove_mask |= equal_mask  # OR 연산을 통해 해당 정점을 마스킹
        
    # 디버깅: True인 정점 개수 출력
    true_count = groove_mask.sum().item()
    print(f"🛠️ Groove Mask Debug: {true_count}개의 정점이 홈 영역으로 설정됨.")
    return groove_mask

#---------------------------------------------------------------------------

#고정된 위치에 적대적 객체 위치
def get_place_fix(images, bboxes):
    fix_place_list = ["nw", "ne", "sw", "se", "n", "s", "w", "e"]
    areas_choose = [[] for _ in range(len(images))] #객체의 위치를 저장할 리스트 len(images)가 3이면 [[],[],[]] 이렇게 3개가 생김
    for i in range(args.patch_count):
        places = stick.cal_stick_place_rcnn(bboxes, args.patch_size, args.patch_size, 0.25, fix_place_list[i]) #각 패치별로 nw,ne,sw...등의 위치에 bbox 좌표근처에 적절한 위치를 할당함
        for j in range(len(places)):
            areas_choose[j].append(places[j])
            
    return areas_choose

#강화학습으로 객체의 위치 지정
def get_place_reinforce(images, bboxes, group, faces, trainer):
    actor = rl_utils.Actor(args.patch_count).cuda()
    actor.train()
    actor_optim = optim.Adam(actor.parameters(), lr=9e-4)

    areas = stick.get_stick_area_rcnn(bboxes, args.patch_size, args.patch_size)
    area_lens = torch.FloatTensor([len(elm) for elm in areas]).unsqueeze(1)

    pad = nn.ZeroPad2d(args.patch_size)
    group_clamp = torch.clamp(group, 0, 1)

    # use X-ray renderer to convert a 3D object to an X-ray image
    rend_group = []
    for pt in range(args.patch_count):
        depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
        # simulate function needs a 4-dimension input
        rend, mask = renderer.simulate(depth_img, args.patch_material)
        rend[~mask] = 1
        rend_group.append(rend)

    # Using running mean/std to stablize the reward
    reward_ms = rl_utils.RunningMeanStd(shape=(1,), device="cuda:0")

    last_reward = 0
    ori_shape = [e.shape for e in images]

    for rep in range(200):
        print("RL phase", rep + 1)
        # sample actions
        action_logits = []
        for s in range(len(images)):
            # resize(images[s], (224, 224))
            action_logits.append(actor(images[s].unsqueeze(0)))
        action_logits = torch.cat(action_logits, dim=0)
        dist = Categorical(logits=action_logits)
        actions = dist.sample()

        places = (area_lens * actions.detach().cpu() / 50).floor().long()
        areas_choose = []
        for bi in range(places.shape[0]):
            area_choose = []
            for pi in range(places.shape[1]):
                area_choose.append(areas[bi][places[bi, pi]])
            areas_choose.append(area_choose)

        # get rewards
        images_delta = [e.clone().detach() for e in images]
        images_delta = [pad(e) for e in images_delta]
        for pt in range(args.patch_count):
            rend = rend_group[pt]
            for s in range(len(images_delta)):
                u, v = areas_choose[s][pt]
                images_delta[s][:, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend[0])

        images_cutpad = []
        for s in range(len(images_delta)):
            images_cutpad.append(images_delta[s][:, args.patch_size:ori_shape[s][1]+args.patch_size, args.patch_size:ori_shape[s][2]+args.patch_size])

        rewards = []
        with torch.no_grad():
            for s in range(len(images_cutpad)):
                trainer.reset_meters()
                loss_cls = trainer.forward(images_cutpad[s].unsqueeze(0), bboxes[s].unsqueeze(0), labels[s].unsqueeze(0), scales[s])
                rewards.append(loss_cls.rpn_cls_loss + loss_cls.roi_cls_loss)
        rewards = torch.stack(rewards, dim=0).unsqueeze(1).detach()

        if actions.shape[-1] > 1:
            reward_penal = actions.float().std(dim=-1, keepdim=True)
        else:
            reward_penal = torch.tensor([0.]).cuda()
        print(rewards.mean().item(), reward_penal.mean().item())
        rewards += 0.01 * reward_penal

        # early stopping
        cur_reward = rewards.mean().item()
        if cur_reward == last_reward:
            break
        last_reward = cur_reward
            
        # standarize rewards
        reward_ms.update(rewards)
        rewards = (rewards - reward_ms.mean) / torch.sqrt(reward_ms.var)

        # learn
        log_prob = dist.log_prob(actions)
        loss = -(rewards * log_prob).mean()

        actor_optim.zero_grad()
        loss.backward()
        actor_optim.step()

        actions = actions.float()

    return areas_choose


def attack(images, bboxes, labels, net, trainer): #(이미지들,bbox정보, 정답 클래스 라벨,모델,학습을 시키는 trainer 객체)
    """
    Main attack function.
    """
    net.phase = "train" #탐지 모델을 훈련모드로 변환
    images = [e.cuda() for e in images] 
    bboxes = [b.cuda() for b in bboxes]
    labels = [l.cuda() for l in labels] #Gpu로 이동

    #홈 부분의 정점들 로드
    groove_coords = load_groove_coords("groove_points.txt")  # (M, 3)
    
    # create a group of patch objects which have same faces
    # we only optimize the coordinate of vertices
    # but not to change the adjacent relation
    group = []
    for _ in range(args.patch_count):
        vertices, faces = renderer.load_from_file(args.obj_path) #3D object의 점과 면을 불러옴
        groove_mask = create_groove_mask(vertices, groove_coords)
        group.append(vertices.unsqueeze(0))
    
    adj_ls = renderer.adj_list(vertices, faces) #vertices를 연결하는 line
    
    # the shape of group: [patch_count, 3, vertices_count]
    group = torch.cat(group, dim=0).cuda() #group는 3D 적대적 객체들의 vertices가 들어있음
    group_ori = group.clone().detach() #원본 3D객체를 저장
    depth_patch = torch.zeros((1, args.patch_count, args.patch_size, args.patch_size)).uniform_().cuda()

    #최적의 위치 계산 (fix인지 강화학습인지 결정)
    if not args.patch_place == "fix_patch": #강화학습으로 설정한 경우
        group.requires_grad_(True) #group을 학습 대상으로 지정
        optimizer = optim.Adam([group], lr=args.lr) #group을 최적화 대상으로 지정
    else:
        depth_patch.requires_grad_(True)
        optimizer = optim.Adam([depth_patch], lr=args.lr)
    # we need a pad function to prevent that a part of patch is out of the image
    pad = nn.ZeroPad2d(args.patch_size)
    
    print("Calculate best place before attack...")

    if args.patch_place == "fix" or args.patch_place == "fix_patch":
        areas_choose = get_place_fix(images, bboxes)
    elif args.patch_place == "reinforce":
        areas_choose = get_place_reinforce(images, bboxes, group, faces, trainer)

    print("Attacking...")
    ori_shape = [e.shape for e in images] #원본 이미지의 크기르 되돌리기 위함
    for t in range(args.num_iters):
        timer = time.time()
        #공격을 하기 전 원본 이미지의 복사본인 images_delta를 만들고 delta에 공격을 가함
        images_delta = [e.clone().detach() for e in images]
        images_delta = [pad(e) for e in images_delta] #패딩 추가
        
        # calculate the perspective loss
        #지각 손실을 계산하여 패치가 너무 변형되지 않게 함
        loss_per = torch.zeros((1,)).cuda() #지각손실함수
        if not args.patch_place == "fix_patch": #fix_patch이면 고정된형태를 유지하므로 지각손실(tv)를 계산할 필요가 없음
            # TV + Shape 유사성 손실을 합쳐서 계산
            tv_sum = 0.0
            shape_sum = 0.0
            
            for pt in range(args.patch_count): #tv loss 계산
                # (1) 기존 TV 손실 (스파이크 억제)
                tv_val = renderer.tvloss(group_ori[pt], group[pt], adj_ls, coe=0)
                tv_sum += tv_val
                
                # (2) Shape 유사성 손실 (원본 열쇠 형태 유지)
                shape_val = shape_loss(
                    group_ori[pt], 
                    group[pt],
                    z_weight=2.0,
                    mask = groove_mask
                    )
                
                shape_sum += shape_val
                
            # patch_count로 평균
            tv_sum /= args.patch_count
            shape_sum /= args.patch_count
            
            # 원하는 비율(γ)로 두 손실을 합산
            gamma = 0.7  
            loss_per = tv_sum + gamma * shape_sum
        
        # clamp the group into [0, 1]
        #값을 0~1사이로 정규화
        group_clamp = torch.clamp(group, 0, 1)
        depth_clamp = torch.clamp(depth_patch, 0, 1)
        
        # use X-ray renderer to convert a 3D object to an X-ray image
        '''
        패치를 x-ray화 하는 법
        1)우선 3D 패치의 depth이미지를 구함 (2D 이미지인데 z의 값에 따라 즉, 깊이에 따라 각 픽셀의 강도를 나타내는 이미지)
        2)depth iamge를 x-ray에 투영
        3)x-ray화 된 패치를 이미지에 삽입
        '''
        for pt in range(args.patch_count):
            if not args.patch_place == "fix_patch": 
                #depth 이미지 구하기
                depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
            else:
                depth_img = depth_clamp[:, pt:pt+1]
            # simulate function needs a 4-dimension input
            #depth image를 x-ray화
            rend, mask = renderer.simulate(depth_img, args.patch_material.replace("_fix", ""))
            rend[~mask] = 1
            #x-ray 이미지에 패치를 적용
            for s in range(len(images_delta)):
                u, v = areas_choose[s][pt] #패치를 적용할 위치 u,v
                # print(rend.shape, images_delta[s].shape)
                # print(u, v, bboxes[s])
                images_delta[s][:, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend[0])
        #패딩을 제거하여 이미지를 원본 크기로 되돌림
        images_cutpad = [] 
        for s in range(len(images_delta)):
            images_cutpad.append(images_delta[s][:, args.patch_size:ori_shape[s][1]+args.patch_size, args.patch_size:ori_shape[s][2]+args.patch_size]) #패치를 이미지에 적용
        # out = net(images_delta[:, :, args.patch_size:300+args.patch_size, args.patch_size:300+args.patch_size])
        
        #객체탐지손실 계산
        loss = 0
        for s in range(len(images_cutpad)):
            trainer.reset_meters()
            loss_cls = trainer.forward(images_cutpad[s].unsqueeze(0), bboxes[s].unsqueeze(0), labels[s].unsqueeze(0), scales[s]) #cls 손실함수들을 구해옴
            loss += loss_cls.rpn_cls_loss + loss_cls.roi_cls_loss #faster-rcnn의 손실함수들인 roi,rpn cls_loss를 가져옴
        loss /= len(images_cutpad) #손실함수의 평균
        
        #최종손실계산 및 역전파
        loss_adv = - loss #적대적 공격을 수행하므로 부호를 반대로 바꿔줌 (적대적 공격이니깐 손실함수가 클수록 좋으므로)
        loss_total = loss_adv + args.beta * loss_per #지각손실함수를 추가하여 패치가 너무 왜곡되지 않도록 함 (이제 최종 손실함수가 만들어짐)
       
        #최적화 (패치 업데이트)
        optimizer.zero_grad()
        loss_total.backward() #역전파 수행
        if not args.patch_place == "fix_patch":
            inan = group.grad.isnan()
            group.grad.data[inan] = 0
            
            # 홈 이외 부분은 grad=0
            #group.grad[:, ~groove_mask, :] = 0
            
            # Y축을 제외한 나머지 축의 그래디언트를 0으로 설정
            group.grad[..., [0,2]] = 0  # X축(0)과 Z축(2)의 그래디언트를 0으로
            
        optimizer.step()
        '''
        group을 텐서로 변형시켜서 훈련할 수 있게 만듬
        group은 적대적 객체에 대한 3D 좌표 (vertex)를 가지고 있음
        ex)
        group = [  [[0.1, 0.2, 0.3],  #적대적 객체 1
                    [0.4, 0.5, 0.6], 
                    [0.7, 0.8, 0.9], 
                    [0.2, 0.3, 0.4], 
                    [0.5, 0.6, 0.7]], 

                    [[0.2, 0.3, 0.4], #적대적 객체 2
                    [0.5, 0.6, 0.7], 
                    [0.8, 0.9, 1.0], 
                    [0.3, 0.4, 0.5], 
                    [0.6, 0.7, 0.8]]] 
        대충 이런식으로 구성되어 있어서 마치 W 마냥 업데이트가 가능한 것            
        ''' 
        torch.cuda.synchronize()
        
        print("Iter: {}/{}, L_adv = {:.3f}, βL_per = {:.3f}, Total loss = {:.3f}, Time: {:.2f}".format(
            t+1, args.num_iters, loss_adv.item() * 1000, args.beta * loss_per.item() * 1000,
            loss_total.item() * 1000, time.time() - timer))
            
    print("Calculate best place after attack...")
    '''
    이전과정들은 일단 위치를 지정하고 해당 위치에서 공격을 위한 패치를 제작
    이후에는 제작된 패치를 가지고 더 효과적인 위치를 찾음 
    흠 근데 이 부분은 reinforce 위치선택을 위해 존재하지 않나 싶음
    '''
    group_clamp = torch.clamp(group, 0, 1) #group (적대적 객체)를 0~1사이로 범위 재조정
    depth_clamp = torch.clamp(depth_patch, 0, 1)
    images_adv = [pad(e).clone().detach() for e in images] #패치를 적용할 원본 이미지
    for pt in range(args.patch_count):
        if not args.patch_place == "fix_patch":
            depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
        else:
            depth_img = depth_clamp[:, pt:pt+1]
        # simulate function needs a 4-dimension input
        rend, mask = renderer.simulate(depth_img, args.patch_material)
        rend[~mask] = 1
        for s in range(len(images_adv)):
            u, v = areas_choose[s][pt]
            images_adv[s][:, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend[0])

    images_cutpad = []
    for s in range(len(images_adv)):
        images_cutpad.append(images_adv[s][:, args.patch_size:ori_shape[s][1]+args.patch_size, args.patch_size:ori_shape[s][2]+args.patch_size])
        
    return images_cutpad, areas_choose, torch.clamp(group, 0, 1), faces
    '''
    패치가 적용된 이미지,
    최적의 위치,
    패치의 모양 return
    '''

def save_img(path, img_tensor, shape):
    img_tensor = img_tensor.cpu().detach().numpy().astype(np.uint8)
    img = img_tensor.transpose(1, 2, 0)
    img = cv2.resize(img, (shape[1], shape[0]))
    cv2.imwrite(path, img)

if __name__ == "__main__":
    if args.dataset == "OPIXray":
        data_info = config.OPIXray_test
    elif args.dataset == "HiXray":
        data_info = config.HiXray_test

    num_classes = len(data_info["model_classes"]) + 1
    net = FasterRCNNVGG16(config.FasterRCNN, num_classes - 1)
    
    '''
    병렬학습 부분 수정
    '''
    state_dict = torch.load(args.ckpt_path)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)

    net.cuda()
    #net.load_state_dict(torch.load(args.ckpt_path))
    
    gpu_count = torch.cuda.device_count()
    print("CUDA is available:", torch.cuda.is_available())
    print("CUDA visible device count:", gpu_count)
    
    if gpu_count > 1:
        print(f"Using {gpu_count} GPUs via DataParallel!")
        net = nn.DataParallel(net, device_ids=[0, 1])  # GPU 0, 1 사용
        
    trainer = FasterRCNNTrainer(net, config.FasterRCNN, num_classes).cuda()
    net.eval()

    dataset = RCNNDetectionDataset(root=data_info["dataset_root"], 
                               model_classes=data_info["model_classes"],
                               image_sets=data_info["imagesetfile"], 
                               target_transform=RCNNAnnotationTransform(data_info["model_classes"]), 
                               phase='test')
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=rcnn_detection_collate_attack, pin_memory=True)

    num_images = len(dataset)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    img_path = os.path.join(args.save_path, "adver_image")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
        
    obj_path = os.path.join(args.save_path, "adver_obj")
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

    for i, (images, bboxes, labels, scales, img_ids) in enumerate(data_loader):
        print("Batch {}/{}...".format(i+1, math.ceil(num_images / args.batch_size)))
        print(img_ids)
        if args.patch_place != "none":
            images_adv, areas_choose, vertices, faces =  attack(images, bboxes, labels, net, trainer)
        else:
            images_adv = images
            faces = None
        
        print("Saving...")
        for t in range(len(images_adv)):
            shape = images_adv[t].shape[1:]
            shape = [int(shape[0] / scales[t]), int(shape[1] / scales[t])]
            save_img(os.path.join(img_path, img_ids[t] + ".png"), images_adv[t] * 255, shape)
            if faces is not None:
                for i in range(vertices.shape[0]):
                    renderer.save_to_file(
                        os.path.join(obj_path, str(img_ids[t]) + "_u{}_v{}.obj".format(*areas_choose[t][i])), 
                        vertices[i], faces)