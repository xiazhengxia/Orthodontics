import os
import argparse
import numpy as np

import torch

from TeethDataset import visualize_colored_point_cloud, augment_insert_asymmetric_inference, cal_norm, cal_norm_global


seed = 1010101
np.random.seed(seed=seed)#随机种子，这样每次运行结果一致
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda:0,1,2,3')
parser.add_argument('--test_dir', type=str, default=r"data/ISICDM-ATRC-test-phase/atrc_test.data",
                    help='test data path')
parser.add_argument('--save_dir', type=str, default=r"result/best_trainval",
                    help='result path')
parser.add_argument('--checkpoints', type=str, default=r"logs/checkpoints/best_trainval.pth", help='checkpoints path')
parser.add_argument('--test_idx', type=int, default=-1, help='The specified test id')
args = parser.parse_args()

def load_test_data(data_path):
    return np.load(data_path)

def preprocess(src_cloud):
    temp_cloud = src_cloud.copy()
    temp_cloud = temp_cloud.reshape((32, 128, 3))
    empty_idx = [np.all(temp_cloud[i] == 0) for i in range(32)]

    aug_cloud = augment_insert_asymmetric_inference(src_cloud)
    # norm_cloud, norm_center, norm_scale = cal_norm(aug_cloud)
    norm_cloud, norm_center, norm_scale = cal_norm_global(aug_cloud)
    return norm_cloud, norm_center, norm_scale, empty_idx

def postprocess(src_cloud, pred, empty_idx):
    src_cloud = src_cloud.reshape(32, 128, 3)
    dst_cloud = np.zeros_like(src_cloud)
    pred = pred.reshape(32, 4, 4)
    for i in range(32):
        if empty_idx[i]:
            dst_cloud[i] = np.zeros_like(dst_cloud[i])
            pred[i] = np.zeros_like(pred[i])
        else:
            src_points = np.hstack((src_cloud[i], np.ones((128, 1)))).transpose(1, 0)
            dst_cloud[i] = np.matmul(pred[i], src_points).transpose(1, 0)[:, :3]
    dst_cloud = dst_cloud.reshape(-1, 3)
    return dst_cloud, pred




def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device(args.device)

    # 加载测试数据
    test_data = load_test_data(args.test_dir)

    # 加载模型
    model = torch.load(args.checkpoints, map_location=device, weights_only=False)
    model.to(device)
    
    if args.test_idx == -1:
        result = []
        for i in range(test_data.shape[0]):
            model.eval()
            with torch.no_grad():
                src_cloud = test_data[i]
                norm_cloud, norm_center, norm_scale, empty_idx = preprocess(src_cloud)

                input_cloud = torch.unsqueeze(torch.from_numpy(norm_cloud), 0).float().to(device)

                pred, _, _ = model(input_cloud)

                pred_cloud, pred_trans = postprocess(src_cloud, pred.detach().cpu().numpy(), empty_idx)
                result.append(pred_trans)

                src_save_path = os.path.join(args.save_dir, f"{i}_src.png")
                dst_save_path = os.path.join(args.save_dir, f"{i}_dst.png")
                visualize_colored_point_cloud(src_cloud.reshape(-1, 128, 3), f"原始点云_{i}", src_save_path)
                visualize_colored_point_cloud(pred_cloud.reshape(-1, 128, 3), f"矫正点云_{i}", dst_save_path)
        result = np.array(result)
        np.save("data/ISICDM-ATRC-test-phase/atrc_test_R.solution", result)
    else:
        model.eval()
        with torch.no_grad():
            src_cloud = test_data[args.test_idx]
            norm_cloud, norm_center, norm_scale, empty_idx = preprocess(src_cloud)

            input_cloud = torch.unsqueeze(torch.from_numpy(norm_cloud), 0).float().to(device)

            pred, _, _ = model(input_cloud)

            pred_cloud, pred_trans = postprocess(src_cloud, pred.detach().cpu().numpy(), empty_idx)

            visualize_colored_point_cloud(src_cloud.reshape(-1, 128, 3), f"原始点云_{args.test_idx}")
            visualize_colored_point_cloud(pred_cloud.reshape(-1, 128, 3), f"矫正点云_{args.test_idx}")
        
    print("finished")
    return

if __name__ == '__main__':
    main(args)

