import cv2
import config
from scipy.optimize import least_squares
import numpy as np

##########################
# 两张图之间的特征提取及匹配
##########################
def extract_features(image_names):
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
    key_points_for_all = []
    descriptor_for_all = []
    colors_for_all = []
    for i, image_name in enumerate(image_names):
        print(i)
        image = cv2.imread(image_name)

        if image is None:
            continue
        key_points, descriptor = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

        if len(key_points) <= 20:
            continue

        key_points_for_all.append(key_points)
        descriptor_for_all.append(descriptor)
        colors = np.zeros((len(key_points), 3))
        for i, key_point in enumerate(key_points):
            p = key_point.pt
            colors[i] = image[int(p[1])][int(p[0])]
        colors_for_all.append(colors)
    return np.array(key_points_for_all), np.array(descriptor_for_all), np.array(colors_for_all)


def match_features(query, train):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(query, train, k=2)
    matches = []

    for m, n in knn_matches:
        if m.distance < config.MRT * n.distance:
            matches.append(m)

    return np.array(matches)


def match_all_features(descriptor_for_all):
    matches_for_all = []
    for i in range(len(descriptor_for_all) - 1):
        print(i)
        matches = match_features(descriptor_for_all[i], descriptor_for_all[i + 1])
        matches_for_all.append(matches)
    return np.array(matches_for_all)


######################
# 寻找图与图之间的对应相机旋转角度以及相机平移
######################
def find_transform(K, p1, p2):
    focal_length = 0.5 * (K[0, 0] + K[1, 1])
    principle_point = (K[0, 2], K[1, 2])
    E, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    cameraMatrix = np.array([[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, cameraMatrix, mask)

    return R, T, mask


def get_matched_points(p1, p2, matches):
    src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
    dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])

    return src_pts, dst_pts


def get_matched_colors(c1, c2, matches):
    color_src_pts = np.asarray([c1[m.queryIdx] for m in matches])
    color_dst_pts = np.asarray([c2[m.trainIdx] for m in matches])

    return color_src_pts, color_dst_pts


# 选择重合的点
def maskout_points(p1, mask):
    p1_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p1_copy.append(p1[i])

    return np.array(p1_copy)


def init_structure(K, key_points_for_all, colors_for_all, matches_for_all):
    p1, p2 = get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0])
    c1, c2 = get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0])

    if find_transform(K, p1, p2):
        R, T, mask = find_transform(K, p1, p2)
    else:
        R, T, mask = np.array([]), np.array([]), np.array([])

    p1 = maskout_points(p1, mask)
    p2 = maskout_points(p2, mask)
    colors = maskout_points(c1, mask)
    # 设置第一个相机的变换矩阵，即作为剩下摄像机矩阵变换的基准。
    R0 = np.eye(3, 3)
    T0 = np.zeros((3, 1))
    structure = reconstruct(K, R0, T0, R, T, p1, p2)
    rotations = [R0, R]
    motions = [T0, T]
    correspond_struct_idx = []
    for key_p in key_points_for_all:
        correspond_struct_idx.append(np.ones(len(key_p)) * - 1)
    correspond_struct_idx = np.array(correspond_struct_idx)
    idx = 0
    matches = matches_for_all[0]
    for i, match in enumerate(matches):
        print(i)
        if mask[i] == 0:
            continue
        correspond_struct_idx[0][int(match.queryIdx)] = idx
        correspond_struct_idx[1][int(match.trainIdx)] = idx
        idx += 1
    return structure, correspond_struct_idx, colors, rotations, motions


#############
# 三维重建
#############
def reconstruct(K, R1, T1, R2, T2, p1, p2):
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))
    proj1[0:3, 0:3] = np.float32(R1)
    proj1[:, 3] = np.float32(T1.T)
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3] = np.float32(T2.T)
    fk = np.float32(K)
    proj1 = np.dot(fk, proj1)
    proj2 = np.dot(fk, proj2)
    s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
    structure = []

    for i in range(len(s[0])):
        col = s[:, i]
        col /= col[3]
        structure.append([col[0], col[1], col[2]])

    return np.array(structure)


###########################
# 将已作出的点云进行融合
###########################
def fusion_structure(matches, struct_indices, next_struct_indices, structure, next_structure, colors, next_colors):
    for i, match in enumerate(matches):
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis=0)
        colors = np.append(colors, [next_colors[i]], axis=0)
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1
    return struct_indices, next_struct_indices, structure, colors


# 制作图像点以及空间点
def get_objpoints_and_imgpoints(matches, struct_indices, structure, key_points):
    object_points = []
    image_points = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx < 0:
            #print("struct_id: ", struct_idx)
            continue
        #print("struct_id: ", struct_idx)
        object_points.append(structure[int(struct_idx)])
        image_points.append(key_points[train_idx].pt)

    return np.array(object_points), np.array(image_points)


########################
# bundle adjustment
########################

def get_3dpos(pos, ob, r, t, K):
    dtype = np.float32

    def F(x):
        p, J = cv2.projectPoints(x.reshape(1, 1, 3), r, t, K, np.array([]))
        p = p.reshape(2)
        e = ob - p
        err = e

        return err

    res = least_squares(F, pos)
    return res.x


def get_3dpos_v1(pos, ob, r, t, K):
    p, J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
    p = p.reshape(2)
    e = ob - p
    if abs(e[0]) > config.x or abs(e[1]) > config.y:
        return None
    return pos


def bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure):
    for i in range(len(rotations)):
        r, _ = cv2.Rodrigues(rotations[i])
        rotations[i] = r
    for i in range(len(correspond_struct_idx)):
        point3d_ids = correspond_struct_idx[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]
        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            new_point = get_3dpos_v1(structure[point3d_id], key_points[j].pt, r, t, K)
            structure[point3d_id] = new_point

    return structure
