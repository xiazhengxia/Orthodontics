import os.path
import numpy as np

import pyvista as pv
import vtk
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import random
random.seed(1010101)

import logging
pyvista_logger = logging.getLogger('pyvista')
pyvista_logger.setLevel(logging.CRITICAL)
output_window = vtk.vtkOutputWindow()
# 将其设置为无实例，即禁用所有输出
output_window.SetInstance(None)


class TeethDataset(Dataset):
    def __init__(self, data_dir, transforms, mode="train"):
        super(TeethDataset, self).__init__()

        self.mode = mode
        self.clouds = None
        self.labels = None
        if mode == "train":
            self.clouds = np.load(os.path.join(data_dir, r"ISICDM-ATRC-develop-phase/atrc_train.data"))
            self.labels = np.load(os.path.join(data_dir, r"ISICDM-ATRC-develop-phase/atrc_train.solution"))
        elif mode == "trainval":
            train_clouds = np.load(os.path.join(data_dir, r"ISICDM-ATRC-develop-phase/atrc_train.data"))
            train_labels = np.load(os.path.join(data_dir, r"ISICDM-ATRC-develop-phase/atrc_train.solution"))
            val_clouds = np.load(os.path.join(data_dir, r"ISICDM-ATRC-develop-phase/atrc_val.data"))
            val_labels = np.load(os.path.join(data_dir, r"ISICDM-ATRC-develop-phase/atrc_val.solution"))
            self.clouds = np.concatenate((train_clouds, val_clouds), axis=0)
            self.labels = np.concatenate((train_labels, val_labels), axis=0)
        elif mode == "val":
            self.clouds = np.load(os.path.join(data_dir, r"ISICDM-ATRC-develop-phase/atrc_val.data"))
            self.labels = np.load(os.path.join(data_dir, r"ISICDM-ATRC-develop-phase/atrc_val.solution"))
        else:
            self.clouds = np.load(os.path.join(data_dir, r"ISICDM-ATRC-test-phase/atrc_test.data"))
            self.labels = np.load(os.path.join(data_dir, r"ISICDM-ATRC-test-phase/atrc_test_R.solution"))

        
        assert self.clouds.shape[0] == self.labels.shape[0], "cloud and label shapes not match"
        print(f"{mode} data loaded: clouds shape {self.clouds.shape}, labels shape {self.labels.shape}")

        self.transforms = DataAugment(transforms, mode=mode)

        # 固定参数
        self.num_teeth = 32
        self.num_points = 128
        
        # print(f"{mode} std: \n", np.std(self.labels, axis=(0, 1)))
        # print(f"{mode} mean: \n", np.mean(self.labels, axis=(0, 1)))

    def __len__(self):
        return self.clouds.shape[0]

    def __getitem__(self, index):
        src_cloud = self.clouds[index]
        label = self.labels[index]
        dst_cloud = self.__cloud_src2dst(src_cloud, label)

        # 记录无数据点云索引
        src_temp = src_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        empty_idx = [np.all(src_temp[i] == 0) for i in range(32)]

        src_cloud, dst_cloud, label = self.transforms(src_cloud, dst_cloud, label)

        # 去除无数据点云
        src_cloud = src_cloud.reshape((self.num_teeth, self.num_points, 3))
        dst_cloud = dst_cloud.reshape((self.num_teeth, self.num_points, 3))
        # label = label.reshape((self.num_teeth, 4, 4))
        for i in range(self.num_teeth):
            if empty_idx[i]:
                src_cloud[i] = np.zeros_like(src_cloud[i])
                dst_cloud[i] = np.zeros_like(dst_cloud[i])
                label[i] = np.eye(4)
        src_cloud = src_cloud.reshape((self.num_teeth * self.num_points, 3))
        dst_cloud = dst_cloud.reshape((self.num_teeth * self.num_points, 3))

        # norm_cloud, norm_center, norm_scale = self.__cal_norm(src_cloud)
        norm_cloud, norm_center, norm_scale = self.__cal_norm_global(src_cloud)
        # print(np.mean(src_cloud), np.max(src_cloud), np.min(src_cloud))
        # print(np.mean(norm_cloud), np.max(norm_cloud), np.min(norm_cloud))
        return norm_cloud, dst_cloud, label, norm_center, norm_scale
    
    def __cloud_src2dst(self, src_cloud, label):
        def points_src2dst(src_points, m):
            src_points_h = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
            dst_points = np.matmul(m, src_points_h.transpose(1, 0)).transpose(1, 0)
            return dst_points[:, :3]
        
        dst_cloud = np.copy(src_cloud)
        dst_cloud = dst_cloud.reshape((self.num_teeth, self.num_points, 3))
        for i in range(self.num_teeth):
            dst_cloud[i] = points_src2dst(dst_cloud[i], label[i])

        dst_cloud = dst_cloud.reshape((self.num_teeth * self.num_points, 3))
        return dst_cloud

    def __cal_norm(self, src_cloud):
        norm_cloud = src_cloud.copy()
        norm_cloud = norm_cloud.reshape((self.num_teeth, self.num_points, 3))
        norm_center = np.zeros((self.num_teeth, 3))
        norm_scale = np.ones((self.num_teeth, 1))
        for i in range(self.num_teeth):
            centroid = np.mean(norm_cloud[i], axis=0)
            norm_cloud[i] = norm_cloud[i] - centroid
            scale = np.max(np.sqrt(np.sum(norm_cloud[i]**2, axis=1)))
            if scale == 0:
                continue
            norm_cloud[i] = norm_cloud[i] / scale
            norm_center[i] = centroid
            norm_scale[i] = scale

        norm_cloud = norm_cloud.reshape((self.num_teeth * self.num_points, 3))
        return norm_cloud, norm_center, norm_scale

    def __cal_norm_global(self, src_cloud):
        pc = src_cloud.copy()
        centroid = np.mean(src_cloud, axis=0)
        pc = pc - centroid
        scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / scale
        return pc, centroid, scale
    

class DataAugment:
    def __init__(self, methods: list, p: float = 0.2, mode="train"):
        self.methods = methods
        self.p = p
        # 固定参数
        self.num_teeth = 32
        self.num_points = 128
        self.mode = mode

    def __call__(self, src_cloud, dst_cloud, label):
        augment_methods = []
        if self.mode == "train" or self.mode == "trainval":
            for i in range(len(self.methods)):
                value = random.random()
                if value < self.p:
                    augment_methods.append(self.methods[i])
            augment_methods.append("augment_insert_asymmetric")
            augment_methods.append("augment_point_shuffle")
        else:
            augment_methods.append("augment_insert_asymmetric")
        # print(augment_methods)

        src_cloud_aug, dst_cloud_aug, label_aug = src_cloud.copy(), dst_cloud.copy(), label.copy()

        for augment_method in augment_methods:
            try:
                src_cloud_aug, dst_cloud_aug, label_aug = eval(f"self.{augment_method}(src_cloud_aug, dst_cloud_aug, label_aug)")
            except Exception as e:
                pass

        return src_cloud_aug,  dst_cloud_aug, label_aug

    def augment_rotation(self, src_cloud, dst_cloud, label, p: float = 0.1, angle: float = 20):
        """
        对原始牙齿点云进行随机旋转，以每个牙齿的中心为旋转原点，并更新相关的变换矩阵。

        参数:
            src_cloud: numpy.ndarray, 原始点云数据，形状通常是 (num_teeth * num_points, 3)。
            dst_cloud: numpy.ndarray, 目标点云数据。
            label: numpy.ndarray, 牙齿的变换矩阵，形状通常是 (num_teeth, 4, 4)。
            p: float, 单个牙齿的旋转概率。
            angle: float, 随机旋转角度的范围 (度数)。

        返回:
            src_cloud_aug: numpy.ndarray, 旋转后的原始点云数据。
            dst_cloud_aug: numpy.ndarray, 未改变的目标点云数据。
            label_aug: numpy.ndarray, 更新后的标签/变换矩阵。
        """
        src_cloud_reshaped = src_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        label_aug = label.copy()

        for i in range(self.num_teeth):
            if random.random() < p:
                current_src_points = src_cloud_reshaped[i]
                current_m = label_aug[i]

                center = np.mean(current_src_points, axis=0)

                rx = np.random.uniform(-angle, angle) * np.pi / 180
                ry = np.random.uniform(-angle, angle) * np.pi / 180
                rz = np.random.uniform(-angle, angle) * np.pi / 180

                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]
                ])
                Ry = np.array([
                    [np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]
                ])
                Rz = np.array([
                    [np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]
                ])
                R_combined = Rz @ Ry @ Rx

                src_points_centered = current_src_points - center
                src_points_rotated = src_points_centered @ R_combined
                src_points_aug = src_points_rotated + center

                T_to_origin = np.eye(4)
                T_to_origin[:3, 3] = -center

                R_combined_4x4 = np.eye(4)
                R_combined_4x4[:3, :3] = R_combined

                T_from_origin = np.eye(4)
                T_from_origin[:3, 3] = center

                rotation_around_center_matrix = T_from_origin @ R_combined_4x4 @ T_to_origin
                label_aug[i] = current_m @ rotation_around_center_matrix

                src_cloud_reshaped[i] = src_points_aug

        src_cloud_aug = src_cloud_reshaped.reshape((self.num_teeth * self.num_points, 3))
        dst_cloud_aug = dst_cloud.copy()
        label_aug = label_aug

        return src_cloud_aug, dst_cloud_aug, label_aug

    def augment_translation(self, src_cloud, dst_cloud, label, p: float = 0.1, scale: float = 0.2):
        """
        对原始牙齿点云进行随机平移，并更新相关的变换矩阵。

        参数:
            src_cloud: numpy.ndarray, 原始点云数据，形状通常是 (num_teeth * num_points, 3)。
            dst_cloud: numpy.ndarray, 目标点云数据。
            label: numpy.ndarray, 牙齿的变换矩阵，形状通常是 (num_teeth, 4, 4)。
            p: float, 单个牙齿的平移概率。
            scale: float, 随机平移范围(牙齿本身尺寸的比例)。

        返回:
            src_cloud_aug: numpy.ndarray, 平移后的原始点云数据。
            dst_cloud_aug: numpy.ndarray, 未改变的目标点云数据。
            label_aug: numpy.ndarray, 更新后的标签/变换矩阵。
        """

        src_cloud_reshaped = src_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        label_aug = label.copy() # 对 label 进行深度拷贝，以更新其平移部分

        for i in range(self.num_teeth):
            if random.random() < p:
                current_src_points = src_cloud_reshaped[i]
                current_m = label_aug[i] # 获取当前牙齿的变换矩阵

                # 1. 计算点云在XYZ方向上的尺寸
                len_x = np.max(current_src_points[:, 0]) - np.min(current_src_points[:, 0])
                len_y = np.max(current_src_points[:, 1]) - np.min(current_src_points[:, 1])
                len_z = np.max(current_src_points[:, 2]) - np.min(current_src_points[:, 2])

                # 2. 生成随机平移量
                tx = np.random.uniform(-scale * len_x, scale * len_x)
                ty = np.random.uniform(-scale * len_y, scale * len_y)
                tz = np.random.uniform(-scale * len_z, scale * len_z)
                translation_vector = np.array([tx, ty, tz])

                # 3. 应用平移到 src_cloud
                src_points_aug = current_src_points + translation_vector

                # 4. Create a 4x4 translation matrix
                T_translation = np.eye(4)
                T_translation[:3, 3] = -translation_vector

                # 5. Update the transformation matrix
                # Apply the new translation *after* the existing transformation
                label_aug[i] = current_m @ T_translation

                # 将平移后的点云赋值回 reshaped 数组
                src_cloud_reshaped[i] = src_points_aug

        # 重塑回原始的 (num_teeth * num_points, 3) 形状
        src_cloud_aug = src_cloud_reshaped.reshape((self.num_teeth * self.num_points, 3))

        # 目标点云通常不平移，直接拷贝
        dst_cloud_aug = dst_cloud.copy()

        return src_cloud_aug, dst_cloud_aug, label_aug


    def augment_scaling(self, src_cloud, dst_cloud, label, scale_range: float = 0.2):
        """
        对所有牙齿进行统一的随机尺度缩放，并更新增强前后的点云和旋转平移矩阵。
        缩放发生在世界坐标系中，并确保点云和标签同步一致。

        参数:
            src_cloud: numpy.ndarray, 原始点云数据，形状通常是 (num_teeth * num_points, 3)。
            dst_cloud: numpy.ndarray, 目标点云数据，形状通常是 (num_teeth * num_points, 3)。
            label: numpy.ndarray, 牙齿的变换矩阵，形状通常是 (num_teeth, 4, 4)。
            p: float, 进行缩放的概率（对整个场景统一应用）。
            scale_range: float, 随机缩放范围。例如，如果 scale_range=0.1，则缩放因子在 [1-0.1, 1+0.1] 之间。

        返回:
            src_cloud_aug: numpy.ndarray, 缩放后的原始点云数据。
            dst_cloud_aug: numpy.ndarray, 缩放后的目标点云数据。
            label_aug: numpy.ndarray, 更新后的变换矩阵。
        """
        # 对输入数据进行深度拷贝，确保不修改原始数据
        src_cloud_aug = src_cloud.copy()
        dst_cloud_aug = dst_cloud.copy()
        label_aug = label.copy()

        # 1. 生成一个统一的随机缩放因子
        s = np.random.uniform(1 - scale_range, 1 + scale_range)

        # 2. **直接缩放点云数据**
        # 如果 src_cloud 和 dst_cloud 是世界坐标系中的点，
        # 那么直接乘以缩放因子 s 即可实现相对于世界原点的缩放。
        src_cloud_aug = src_cloud_aug * s
        dst_cloud_aug = dst_cloud_aug * s

        # 3. **更新变换矩阵 label**
        for i in range(self.num_teeth):
            # 保持旋转部分 (3x3) 不变
            # 直接将缩放因子应用于平移部分 (3x1 向量)
            label_aug[i][:3, 3] = label_aug[i][:3, 3] * s

        return src_cloud_aug, dst_cloud_aug, label_aug
    
    def augment_random_erase(self, src_cloud, dst_cloud, label, p: float = 0.1):
        """
        原牙齿与矫正后牙齿随机删除,这个必须放在最后，防止0点被其他增强方法覆盖
        """
        src_cloud_aug = src_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        label_aug = label.copy() # 对 label 进行深度拷贝，以更新其平移部分
        dst_cloud_aug = dst_cloud.copy().reshape((self.num_teeth, self.num_points, 3))

        

        for i in range(self.num_teeth):
            if random.random() < p:
                
                set_idx = None
                if i <=7:
                    set_idx = i
                elif i >= 8 and i <= 15:
                    set_idx = 15 - i
                elif i >= 16 and i <= 23:
                    set_idx = i - 16
                elif i >= 24 and i <= 31:
                    set_idx = 31 - i
                if set_idx is None:
                    continue
                print("1++++++++++++++++++++++++++++++++++++++++++++++++")
                # 获取当前考虑的四个牙齿的全局索引
                upper_left_idx_global = set_idx
                upper_right_idx_global = 15 - set_idx
                lower_left_idx_global = 16 + set_idx # 假设下颌从索引 16 开始
                lower_right_idx_global = 16 + (15 - set_idx) # 假设下颌从索引 16 开始

                # 获取当前四个牙齿的缺失状态
                empty_idx = np.array([np.all(src_cloud_aug[i] == 0) for i in range(self.num_teeth)])
                is_upper_left_empty = empty_idx[upper_left_idx_global]
                is_upper_right_empty = empty_idx[upper_right_idx_global]
                is_lower_left_empty = empty_idx[lower_left_idx_global]
                is_lower_right_empty = empty_idx[lower_right_idx_global]

                
                if is_upper_left_empty + is_upper_right_empty + is_lower_left_empty + is_lower_right_empty < 3:
                    src_cloud_aug[i] = np.zeros_like(src_cloud_aug[i])
                    dst_cloud_aug[i] = np.zeros_like(dst_cloud_aug[i])
                    label_aug[i] = np.eye(4)
                    print("2++++++++++++++++++++++++++++++++++++++++++++++++")
        
        src_cloud_aug = src_cloud_aug.reshape((self.num_teeth * self.num_points, 3))
        dst_cloud_aug = dst_cloud_aug.reshape((self.num_teeth * self.num_points, 3))
        return src_cloud_aug, dst_cloud_aug, label_aug

    def augment_horizontal_flip(self, src_cloud, dst_cloud, label):
        """
        对原始牙齿和矫正后牙齿点云进行水平翻转（沿Y-Z平面镜像，X坐标变号），
        并更新对应的变换矩阵。每个牙齿独立决定是否翻转。

        参数:
            src_cloud: numpy.ndarray, 原始点云数据，形状通常是 (num_teeth * num_points, 3)。
            dst_cloud: numpy.ndarray, 目标点云数据，形状通常是 (num_teeth * num_points, 3)。
            label: numpy.ndarray, 牙齿的变换矩阵，形状通常是 (num_teeth, 4, 4)。
            p: float, 进行翻转的概率。

        返回:
            src_cloud_aug: numpy.ndarray, 翻转后的原始点云数据。
            dst_cloud_aug: numpy.ndarray, 翻转后的目标点云数据。
            label_aug: numpy.ndarray, 更新后的变换矩阵。
        """
        # 对输入数据进行深度拷贝，确保不修改原始数据
        src_cloud_reshaped = src_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        dst_cloud_reshaped = dst_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        label_aug = label.copy()

        # 定义翻转矩阵 (X坐标变号)
        flip_matrix_3x3 = np.diag([-1, 1, 1])  # 仅X轴变号，Y和Z不变
        flip_matrix_4x4 = np.eye(4)
        flip_matrix_4x4[:3, :3] = flip_matrix_3x3

        for i in range(self.num_teeth):
            current_src_points = src_cloud_reshaped[i]
            current_dst_points = dst_cloud_reshaped[i]
            current_label_matrix = label_aug[i]

            # 1. 对当前牙齿的 src_cloud 应用翻转
            # 将点云转换为齐次坐标 (M, 4)
            src_points_homo = np.hstack((current_src_points, np.ones((current_src_points.shape[0], 1))))
            # 应用翻转矩阵到点云 (P_homo @ M.T)
            src_cloud_reshaped[i] = (src_points_homo @ flip_matrix_4x4.T)[:, :3]

            # 2. 对当前牙齿的 dst_cloud 应用翻转 (如果需要)
            dst_points_homo = np.hstack((current_dst_points, np.ones((current_dst_points.shape[0], 1))))
            dst_cloud_reshaped[i] = (dst_points_homo @ flip_matrix_4x4.T)[:, :3]

            # 3. 更新当前牙齿的变换矩阵 label
            # 旋转部分：flip_matrix_3x3 * R * flip_matrix_3x3
            label_aug[i][:3, :3] = flip_matrix_3x3 @ current_label_matrix[:3, :3] @ flip_matrix_3x3
            # 平移部分：flip_matrix_3x3 * t
            label_aug[i][:3, 3] = flip_matrix_3x3 @ current_label_matrix[:3, 3]
        # 将点云数组重塑回扁平的形状
        src_cloud_aug = src_cloud_reshaped.reshape((self.num_teeth * self.num_points, 3))
        dst_cloud_aug = dst_cloud_reshaped.reshape((self.num_teeth * self.num_points, 3))

        return src_cloud_aug, dst_cloud_aug, label_aug

    def augment_point_shuffle(self, src_cloud, dst_cloud, label):
        '''
        对点云进行随机乱序
        '''
        # 对输入数据进行深度拷贝，确保不修改原始数据
        src_cloud_reshaped = src_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        dst_cloud_reshaped = dst_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        label_aug = label.copy()

        shuffled_indices = np.random.permutation(self.num_points)
        src_cloud_reshaped = src_cloud_reshaped[:, shuffled_indices, :]
        dst_cloud_reshaped = dst_cloud_reshaped[:, shuffled_indices, :]

        src_cloud_aug = src_cloud_reshaped.reshape((self.num_teeth * self.num_points, 3))
        dst_cloud_aug = dst_cloud_reshaped.reshape((self.num_teeth * self.num_points, 3))
        return src_cloud_aug, dst_cloud_aug, label_aug
    
    # 辅助函数：计算点云质心
    def _compute_centroid(self, points):
        # 排除零点，避免计算错误，因为0点代表缺失
        valid_points = points[~np.all(points == 0, axis=1)]
        if valid_points.shape[0] == 0:
            return np.array([0., 0., 0.]) # 如果没有有效点，返回原点
        return np.mean(valid_points, axis=0)

    # 新增辅助函数：计算两组点云之间的刚性变换 (R, t)
    def _compute_rigid_transform(self, src_points, dst_points):
        """
        使用 SVD 计算从 src_points 到 dst_points 的最优刚性变换 (R, t)。
        返回 4x4 的齐次变换矩阵。
        src_points 和 dst_points 形状应为 (N, 3)。
        """
        if src_points.shape != dst_points.shape or src_points.shape[0] < 3:
            # 如果点数不足或形状不匹配，返回单位矩阵
            return np.eye(4)

        centroid_src = np.mean(src_points, axis=0)
        centroid_dst = np.mean(dst_points, axis=0)

        centered_src = src_points - centroid_src
        centered_dst = dst_points - centroid_dst

        # 协方差矩阵
        H = centered_src.T @ centered_dst

        U, S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        # 处理反射（如果确定是刚性变换，则行列式必须为正）
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_dst - R @ centroid_src

        # 构建 4x4 齐次变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t

        return transform_matrix

    def augment_insert_asymmetric(self, src_cloud, dst_cloud, label):
        '''
        如果某牙对称位和上下位四个牙中至少有一个存在，则无法整体前移，此时需要对每个缺失位置插入假牙进行补齐，防止前移数据混淆
        If a tooth's symmetrical position or any of its four adjacent (upper/lower/left/right) teeth exists,
        the entire set cannot be shifted forward. In this case, prosthetic teeth need to be inserted
        at each missing position to prevent confusion in forward-shift data.
        '''
        src_cloud_reshaped = src_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        dst_cloud_reshaped = dst_cloud.copy().reshape((self.num_teeth, self.num_points, 3))
        label_aug = label.copy() # label_aug 是一个 (num_teeth, 4, 4) 的矩阵

        empty_idx = np.array([np.all(src_cloud_reshaped[i] == 0) for i in range(self.num_teeth)])
        empty_idx = empty_idx.reshape(2, 16) # 重塑为 (颌骨, 颌骨内牙齿索引)

        # 遍历牙齿索引，从后往前 (7到0)，这通常对应于从磨牙到门牙
        for i in range(7, -1, -1): # Iterates i from 7 down to 0
            # 获取当前考虑的四个牙齿的全局索引
            upper_left_idx_global = i
            upper_right_idx_global = 15 - i
            lower_left_idx_global = 16 + i # 假设下颌从索引 16 开始
            lower_right_idx_global = 16 + (15 - i) # 假设下颌从索引 16 开始

            # 获取当前四个牙齿的缺失状态
            is_upper_left_empty = empty_idx[0][i]
            is_upper_right_empty = empty_idx[0][15 - i]
            is_lower_left_empty = empty_idx[1][i]
            is_lower_right_empty = empty_idx[1][15 - i]

            # 判断是否需要进行填充
            if not (is_upper_left_empty and is_upper_right_empty and is_lower_left_empty and is_lower_right_empty) \
               and (is_upper_left_empty or is_upper_right_empty or is_lower_left_empty or is_lower_right_empty):
                # print("1==================================")
                # --- 辅助函数，用于处理单个缺失牙齿的填充逻辑 ---
                def _fill_missing_tooth(jaw_idx, tooth_in_jaw_idx, global_idx):
                    if not empty_idx[jaw_idx][tooth_in_jaw_idx]:
                        return # 当前牙齿不缺失，无需处理

                    # 尝试寻找左右邻牙
                    left_neighbor_idx_in_jaw = tooth_in_jaw_idx - 1
                    right_neighbor_idx_in_jaw = tooth_in_jaw_idx + 1

                    left_neighbor_exists = False
                    right_neighbor_exists = False
                    
                    neighbor_global_idx = -1 # 用于存储最终选择的邻牙的全局索引

                    # 检查左邻牙
                    if left_neighbor_idx_in_jaw >= 0 and not empty_idx[jaw_idx][left_neighbor_idx_in_jaw]:
                        left_neighbor_exists = True
                        left_neighbor_global_idx = global_idx - 1 

                    # 检查右邻牙
                    if right_neighbor_idx_in_jaw < 16 and not empty_idx[jaw_idx][right_neighbor_idx_in_jaw]:
                        right_neighbor_exists = True
                        right_neighbor_global_idx = global_idx + 1

                    target_centroid = None

                    if left_neighbor_exists and right_neighbor_exists:
                        # 左右邻牙都在，取质心中点作为目标质心
                        left_centroid = self._compute_centroid(src_cloud_reshaped[left_neighbor_global_idx])
                        right_centroid = self._compute_centroid(src_cloud_reshaped[right_neighbor_global_idx])
                        target_centroid = (left_centroid + right_centroid) / 2.0
                        
                        # 选择一个邻牙作为模板，例如左邻牙
                        neighbor_global_idx = left_neighbor_global_idx

                    elif left_neighbor_exists:
                        # 只有左邻牙，目标质心为其自身的质心
                        target_centroid = self._compute_centroid(src_cloud_reshaped[left_neighbor_global_idx])
                        neighbor_global_idx = left_neighbor_global_idx

                    elif right_neighbor_exists:
                        # 只有右邻牙，目标质心为其自身的质心
                        target_centroid = self._compute_centroid(src_cloud_reshaped[right_neighbor_global_idx])
                        neighbor_global_idx = right_neighbor_global_idx
                    else:
                        # 既没有左右邻牙，则无法填充，直接返回
                        return


                    # 如果找到了可以用来填充的邻牙和目标质心
                    if neighbor_global_idx != -1 and target_centroid is not None:
                        # 获取原始邻牙的点云数据和变换矩阵
                        original_neighbor_src_points = src_cloud_reshaped[neighbor_global_idx].copy()
                        original_neighbor_transform = label_aug[neighbor_global_idx].copy()

                        # 确保原始邻牙的 src_points 有效（非零点）
                        valid_original_neighbor_src_points = original_neighbor_src_points[~np.all(original_neighbor_src_points == 0, axis=1)]
                        if valid_original_neighbor_src_points.shape[0] < 3: # 至少需要3个点才能计算质心和变换
                            return # 邻牙数据无效，无法填充

                        # 计算邻牙的当前质心 (在 src_cloud 坐标系下)
                        current_template_centroid = self._compute_centroid(original_neighbor_src_points)

                        # 计算从邻牙当前位置到目标位置的平移向量 (在 src_cloud 坐标系下)
                        translation_vector_src_coords = target_centroid - current_template_centroid

                        # =========================================================
                        # 步骤 1: 获得 new_src 点云 (平移后的邻牙点云)
                        new_src_points = original_neighbor_src_points + translation_vector_src_coords
                        src_cloud_reshaped[global_idx] = new_src_points # 更新 src_cloud_reshaped
                        
                        # 确保 new_src_points 有效
                        valid_new_src_points = new_src_points[~np.all(new_src_points == 0, axis=1)]
                        if valid_new_src_points.shape[0] < 3:
                            # 如果新的 src 点云无效，则不计算其变换
                            label_aug[global_idx] = np.zeros((4, 4))
                            return

                        # 步骤 2: 获得 new_dst 点云
                        # 使用邻牙的原始变换矩阵将 new_src_points 转换到 dst 空间
                        # 需要将点云转换为齐次坐标
                        new_src_points_homog = np.hstack((new_src_points, np.ones((self.num_points, 1))))
                        
                        # 应用邻牙的变换矩阵
                        # 这里我们假设 original_neighbor_transform 是将 original_neighbor_src_points 变换到 original_neighbor_dst_points 的矩阵
                        # 那么它也应该能够将 new_src_points 变换到对应的 new_dst_points
                        new_dst_points_homog = (original_neighbor_transform @ new_src_points_homog.T).T
                        new_dst_points = new_dst_points_homog[:, :3]
                        
                        # 确保 new_dst_points 有效
                        valid_new_dst_points = new_dst_points[~np.all(new_dst_points == 0, axis=1)]
                        if valid_new_dst_points.shape[0] < 3:
                            # 如果新的 dst 点云无效，则不计算其变换
                            label_aug[global_idx] = np.zeros((4, 4))
                            return

                        # 步骤 3: 计算 new_src_points 到 new_dst_points 的新变换矩阵
                        new_transform_matrix = self._compute_rigid_transform(valid_new_src_points, valid_new_dst_points)
                        
                        # 更新 label_aug
                        label_aug[global_idx] = new_transform_matrix

                        temp_points = np.hstack((valid_new_src_points, np.ones((valid_new_src_points.shape[0], 1)))).transpose(1, 0) # 将点云转换为齐次坐标
                        temp_points = np.matmul(new_transform_matrix, temp_points).transpose(1, 0)[:, :3] # 将点云应用变换矩阵
                        dst_cloud_reshaped[global_idx] = temp_points
                        # print("2==================================")
                    else:
                        # 无法找到邻牙或目标质心，该位置的label_aug保持为0或者原始值
                        label_aug[global_idx] = np.zeros((4, 4)) # 或者 np.eye(4)

                # --- 调用辅助函数处理四个可能的缺失位置 ---
                _fill_missing_tooth(0, i, upper_left_idx_global)
                _fill_missing_tooth(0, 15 - i, upper_right_idx_global)
                _fill_missing_tooth(1, i, lower_left_idx_global)
                _fill_missing_tooth(1, 15 - i, lower_right_idx_global)

        # 最后，将重塑后的数据展平回原始形状以返回
        src_cloud_aug = src_cloud_reshaped.reshape((self.num_teeth * self.num_points, 3))
        dst_cloud_aug = dst_cloud_reshaped.reshape((self.num_teeth * self.num_points, 3))
        return src_cloud_aug, dst_cloud_aug, label_aug

def compute_centroid(points):
    # 排除零点，避免计算错误，因为0点代表缺失
    valid_points = points[~np.all(points == 0, axis=1)]
    if valid_points.shape[0] == 0:
        return np.array([0., 0., 0.]) # 如果没有有效点，返回原点
    return np.mean(valid_points, axis=0)

def compute_rigid_transform(src_points, dst_points):
    """
    使用 SVD 计算从 src_points 到 dst_points 的最优刚性变换 (R, t)。
    返回 4x4 的齐次变换矩阵。
    src_points 和 dst_points 形状应为 (N, 3)。
    """
    if src_points.shape != dst_points.shape or src_points.shape[0] < 3:
        # 如果点数不足或形状不匹配，返回单位矩阵
        return np.eye(4)

    centroid_src = np.mean(src_points, axis=0)
    centroid_dst = np.mean(dst_points, axis=0)

    centered_src = src_points - centroid_src
    centered_dst = dst_points - centroid_dst

    # 协方差矩阵
    H = centered_src.T @ centered_dst

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # 处理反射（如果确定是刚性变换，则行列式必须为正）
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_dst - R @ centroid_src

    # 构建 4x4 齐次变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t

    return transform_matrix

def augment_insert_asymmetric_inference(src_cloud):
    '''
    在预测阶段，对缺失牙齿进行补齐，以确保 src_cloud 的结构与训练时一致。
    In inference stage, fills in missing teeth to ensure src_cloud structure
    is consistent with training data.
    '''
    num_teeth = 32
    num_points = 128
    # 确保是副本，避免修改原始输入数据
    src_cloud_reshaped = src_cloud.copy().reshape((num_teeth, num_points, 3))

    # 检查哪些牙齿是缺失的 (所有点都为0)
    empty_idx = np.array([np.all(src_cloud_reshaped[i] == 0) for i in range(num_teeth)])
    empty_idx = empty_idx.reshape(2, 16) # 重塑为 (颌骨, 颌骨内牙齿索引)

    # 遍历牙齿索引，从后往前 (7到0)
    for i in range(7, -1, -1):
        # 获取当前考虑的四个牙齿的全局索引
        upper_left_idx_global = i
        upper_right_idx_global = 15 - i
        lower_left_idx_global = 16 + i # 假设下颌从索引 16 开始
        lower_right_idx_global = 16 + (15 - i) # 假设下颌从索引 16 开始

        # 获取当前四个牙齿的缺失状态
        is_upper_left_empty = empty_idx[0][i]
        is_upper_right_empty = empty_idx[0][15 - i]
        is_lower_left_empty = empty_idx[1][i]
        is_lower_right_empty = empty_idx[1][15 - i]

        # 判断是否需要进行填充：条件与训练时相同
        if not (is_upper_left_empty and is_upper_right_empty and is_lower_left_empty and is_lower_right_empty) \
            and (is_upper_left_empty or is_upper_right_empty or is_lower_left_empty or is_lower_right_empty):

            # --- 辅助函数，用于处理单个缺失牙齿的填充逻辑 (仅修改 src_cloud) ---
            def _fill_missing_tooth_inference(jaw_idx, tooth_in_jaw_idx, global_idx):
                if not empty_idx[jaw_idx][tooth_in_jaw_idx]:
                    return # 当前牙齿不缺失，无需处理

                # 尝试寻找左右邻牙
                left_neighbor_idx_in_jaw = tooth_in_jaw_idx - 1
                right_neighbor_idx_in_jaw = tooth_in_jaw_idx + 1

                left_neighbor_exists = False
                right_neighbor_exists = False
                
                neighbor_global_idx = -1 # 用于存储最终选择的邻牙的全局索引

                # 检查左邻牙
                if left_neighbor_idx_in_jaw >= 0 and not empty_idx[jaw_idx][left_neighbor_idx_in_jaw]:
                    left_neighbor_exists = True
                    left_neighbor_global_idx = global_idx - 1 

                # 检查右邻牙
                if right_neighbor_idx_in_jaw < 16 and not empty_idx[jaw_idx][right_neighbor_idx_in_jaw]:
                    right_neighbor_exists = True
                    right_neighbor_global_idx = global_idx + 1

                target_centroid = None

                if left_neighbor_exists and right_neighbor_exists:
                    # 左右邻牙都在，取质心中点作为目标质心
                    left_centroid = compute_centroid(src_cloud_reshaped[left_neighbor_global_idx])
                    right_centroid = compute_centroid(src_cloud_reshaped[right_neighbor_global_idx])
                    target_centroid = (left_centroid + right_centroid) / 2.0
                    
                    # 选择一个邻牙作为模板，例如左邻牙
                    neighbor_global_idx = left_neighbor_global_idx

                elif left_neighbor_exists:
                    # 只有左邻牙，目标质心为其自身的质心
                    target_centroid = compute_centroid(src_cloud_reshaped[left_neighbor_global_idx])
                    neighbor_global_idx = left_neighbor_global_idx

                elif right_neighbor_exists:
                    # 只有右邻牙，目标质心为其自身的质心
                    target_centroid = compute_centroid(src_cloud_reshaped[right_neighbor_global_idx])
                    neighbor_global_idx = right_neighbor_global_idx
                else:
                    # 既没有左右邻牙，则无法填充，直接返回
                    return

                # 如果找到了可以用来填充的邻牙和目标质心
                if neighbor_global_idx != -1 and target_centroid is not None:
                    # 获取原始邻牙的点云数据
                    original_neighbor_src_points = src_cloud_reshaped[neighbor_global_idx].copy()

                    # 确保原始邻牙的 src_points 有效（非零点）
                    valid_original_neighbor_src_points = original_neighbor_src_points[~np.all(original_neighbor_src_points == 0, axis=1)]
                    if valid_original_neighbor_src_points.shape[0] < 3: # 至少需要3个点才能计算质心
                        return # 邻牙数据无效，无法填充

                    # 计算邻牙的当前质心 (在 src_cloud 坐标系下)
                    current_template_centroid = compute_centroid(original_neighbor_src_points)

                    # 计算从邻牙当前位置到目标位置的平移向量 (在 src_cloud 坐标系下)
                    translation_vector_src_coords = target_centroid - current_template_centroid

                    # 获得 new_src 点云 (平移后的邻牙点云)
                    new_src_points = original_neighbor_src_points + translation_vector_src_coords
                    src_cloud_reshaped[global_idx] = new_src_points # 更新 src_cloud_reshaped
            
            # --- 调用辅助函数处理四个可能的缺失位置 ---
            _fill_missing_tooth_inference(0, i, upper_left_idx_global)
            _fill_missing_tooth_inference(0, 15 - i, upper_right_idx_global)
            _fill_missing_tooth_inference(1, i, lower_left_idx_global)
            _fill_missing_tooth_inference(1, 15 - i, lower_right_idx_global)

    # 最后，将重塑后的数据展平回原始形状以返回
    src_cloud_aug = src_cloud_reshaped.reshape((num_teeth * num_points, 3))
    return src_cloud_aug

def cal_norm( src_cloud):
    num_teeth = 32
    num_points = 128
    norm_cloud = src_cloud.copy()
    norm_cloud = norm_cloud.reshape(num_teeth, num_points, 3)
    norm_center = np.zeros((num_teeth, 3))
    norm_scale = np.ones((num_teeth, 1))
    for i in range(num_teeth):
        centroid = np.mean(norm_cloud[i], axis=0)
        norm_cloud[i] = norm_cloud[i] - centroid
        scale = np.max(np.sqrt(np.sum(norm_cloud[i]**2, axis=1)))
        if scale == 0:
            continue
        norm_cloud[i] = norm_cloud[i] / scale
        norm_center[i] = centroid
        norm_scale[i] = scale

    norm_cloud = norm_cloud.reshape((num_teeth * num_points, 3))
    return norm_cloud, norm_center, norm_scale

def cal_norm_global(src_cloud):
    pc = src_cloud.copy()
    centroid = np.mean(src_cloud, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / scale
    return pc, centroid, scale

def visualize_colored_point_cloud(points, title, save_path=None):
    """
    使用 PyVista 可视化带有不同颜色的三维点云

    参数:
        points: numpy.ndarray, 形状为 (N, M, 3) 的数组，表示 N 个物体，每个物体有 M 个三维点

    返回:
        无，直接显示点云可视化窗口
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("输入必须是 NumPy 数组")

    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("输入数组必须是 (N, M, 3) 的形状")

    N, M, _ = points.shape

    # 分配 N 个不同颜色
    cmap = plt.get_cmap("tab20")  # 可选其他 colormap
    colors = np.zeros((N, M, 3))
    for i in range(N):
        colors[i, :, :] = cmap(i % 20)[:3]  # 忽略 alpha 通道，RGB 颜色

    # 创建 PyVista 的 Plotter 对象
    off_screen = False if save_path is None else True
    plotter = pv.Plotter(window_size=[800, 600], title=title, off_screen=off_screen) # 可以设置窗口大小
    # plotter.set_window_icon() # 设置默认窗口图标

    # 设置窗口标题
    plotter.add_text(title, position='upper_left', color='white', font_size=10)


    # 添加坐标轴（类似 Open3D 的 create_coordinate_frame）
    plotter.add_axes()

    # 设置背景颜色为黑色
    plotter.set_background([255, 255, 255])

    # 合并所有点用于整体点云显示
    all_points = points.reshape(-1, 3)
    all_colors = colors.reshape(-1, 3)

    # 创建点云对象
    point_cloud = pv.PolyData(all_points)
    point_cloud["colors"] = all_colors  # 为点云分配颜色

    # 添加点云到渲染窗口
    plotter.add_points(point_cloud, scalars="colors", rgb=True, point_size=5)

    # 为每个物体尝试生成表面
    for i in range(N):
        pts = points[i]
        # print(f"Object {i}: min = {pts.min():.3f}, max = {pts.max():.3f}, std = {pts.std():.3f}")

        if pts.shape[0] >= 4:  # 确保点数足够
            try:
                # 创建 PyVista 点云对象
                pcd_i = pv.PolyData(pts)

                # 表面重建（使用 PyVista 的 Delaunay 3D 或其他方法）
                mesh = pcd_i.delaunay_3d().extract_surface()

                # 平滑处理（可选，改善表面质量）
                mesh = mesh.smooth(n_iter=100)

                # 计算法向量
                mesh.compute_normals(inplace=True)

                # 为网格分配单一颜色
                mesh_color = cmap(i % 20)[:3]
                mesh["colors"] = np.tile(mesh_color, (mesh.n_points, 1))

                # 添加网格到渲染窗口
                plotter.add_mesh(mesh, scalars="colors", rgb=True, opacity=0.8)

            except Exception as e:
                # print(f"Object {i} 表面生成失败: {e}")
                continue
    if save_path is None:
        plotter.show()
    else:
        plotter.screenshot(save_path)
        plotter.close()
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="src", help='src or dst')
    parser.add_argument('--data_dir', type=str, default=r"data", help='src or dst')
    parser.add_argument('--aug', type=str, nargs='+', 
                        # default=["augment_rotation", "augment_translation", "augment_scaling", "augment_horizontal_flip"], 
                        default=[], 
                        help='single_rotation')
    parser.add_argument('--save_dir', type=str, default=r"result/test",
                        help='result path')
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    train_dataset = TeethDataset(args.data_dir,args.aug, mode="test")
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch_id, (norm_cloud, dst_cloud, label, norm_center, norm_scale) in enumerate(trainDataLoader, 0):
        norm_cloud = norm_cloud.numpy().reshape(-1, 128, 3)
        src_cloud = norm_cloud * norm_scale.numpy().reshape(-1, 1, 1) + norm_center.numpy().reshape(-1, 1, 3)
        print(np.mean(src_cloud), np.max(src_cloud), np.min(src_cloud))

        src_save_path = os.path.join(args.save_dir, f"{batch_id}_src.png")
        dst_save_path = os.path.join(args.save_dir, f"{batch_id}_dst.png")
        visualize_colored_point_cloud(src_cloud.reshape(-1, 128, 3), f"src_{batch_id}", src_save_path)
        visualize_colored_point_cloud(dst_cloud.numpy().reshape(-1, 128, 3), f"dst_{batch_id}", dst_save_path)
  
    #上到下 左到右