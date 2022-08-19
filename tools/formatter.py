import numpy as np


def format_points(points_np, pc_type):
    """
    使用kitti训练的模型使用(xyz,intensity)作为输入;
    使用nuscence训练的模型使用(xyz,intensity, 0)作为输入。
    本方法将输入的点转换为需要的模式。(xyz,intensity)或者(xyz,intensity, 0)
    Args:
        points_np:
        pc_type:(str) kitti or nus
    Returns:
        numpy
        根据模型类型，输出为(xyz,intensity)或者(xyz,intensity, 0)

    """

    if pc_type == 'kitti':
        """
        模型是使用kitti训练的
        input: ("x", "y", "z", "intensity")
        """
        points_np[:, 3] = points_np[:, 3] / 255.0
    elif pc_type == 'nus':
        """
        模型是使用nuscence训练的
        input: ("x", "y", "z", "intensity", 0)
        """
        zeros_np = np.zeros(points_np.shape[0])
        points_np[:, 3] = points_np[:, 3] * 255.0
        points_np = np.insert(points_np, 4, zeros_np, axis=1)

    return points_np


def format_result(result, codebase):
    """
    KITTI模型和nus模型输出的result不同，转成相同的格式
    Args:
        result:mmdet3d.api.inference_detector输出的结果
        codebase: mmdet3d or det3d

    Returns:(dict)：numpy
        'boxes_3d': 3D bbox NX7
        'scores_3d': 置信度 NX1
        'labels_3d': label NX1

    """
    if codebase == 'mmdet3d':
        
        if 'pts_bbox' in result[0].keys():  # nuscence
            boxes_3d = result[0]['pts_bbox']["boxes_3d"].tensor.cpu().detach().numpy()
            scores_3d = result[0]['pts_bbox']["scores_3d"].cpu().detach().numpy().reshape(-1, 1)
            labels_3d = result[0]['pts_bbox']["labels_3d"].cpu().detach().numpy().reshape(-1, 1)

        else:  # kitti
            boxes_3d = result[0]["boxes_3d"].tensor.cpu().detach().numpy()
            scores_3d = result[0]["scores_3d"].cpu().detach().numpy().reshape(-1, 1)
            labels_3d = result[0]["labels_3d"].cpu().detach().numpy().reshape(-1, 1)

    else:  # det3d
        boxes_3d = result[0]["box3d_lidar"].detach().cpu().numpy()
        scores_3d = result[0]["scores"].detach().cpu().numpy().reshape(-1, 1)
        labels_3d = result[0]["label_preds"].detach().cpu().numpy().reshape(-1, 1)

    # det_result = np.column_stack((boxes_3d, labels_3d, scores_3d))  # [0-6, 7, 8]

    det_result = {
        'boxes_3d': boxes_3d,
        'scores_3d': scores_3d.squeeze(-1),
        'labels_3d': labels_3d
    }
    return det_result
