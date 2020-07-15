import os
from utils import *
from sfm import *
from pathplan import *

def SFM():
    imgdir = config.image_dir
    img_names = os.listdir(imgdir)
    img_names = sorted(img_names)

    for i in range(len(img_names)):
        img_names[i] = os.path.join(imgdir, img_names[i])
    print(len(img_names))

    # K是摄像头的参数矩阵
    K = config.K

    # 提取角点，进行角点匹配
    key_points_for_all, descriptor_for_all, colors_for_all = extract_features(img_names)
    matches_for_all = match_all_features(descriptor_for_all)
    structure, correspond_struct_idx, colors, rotations, motions = init_structure(K, key_points_for_all, colors_for_all,
                                                                                  matches_for_all)
    print("匹配： ",matches_for_all.shape)

    for i in range(1, len(matches_for_all)):
        object_points, image_points = get_objpoints_and_imgpoints(matches_for_all[i], correspond_struct_idx[i],
                                                                  structure, key_points_for_all[i + 1])

        print(i, len(image_points))
        # 在python的opencv中solvePnPRansac函数的第一个参数长度需要大于7，否则会报错
        # 这里对小于7的点集做一个重复填充操作，即用点集中的第一个点补满7个
        if len(image_points) < 7:
            while len(image_points) < 7:
                object_points = np.append(object_points, [object_points[0]], axis=0)
                image_points = np.append(image_points, [image_points[0]], axis=0)

        # 点云融合
        _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, K, np.array([]))
        R, _ = cv2.Rodrigues(r)
        rotations.append(R)
        motions.append(T)
        p1, p2 = get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i])
        c1, c2 = get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i])
        next_structure = reconstruct(K, rotations[i], motions[i], R, T, p1, p2)

        correspond_struct_idx[i], correspond_struct_idx[i + 1], structure, colors = fusion_structure(matches_for_all[i],
                                                                                                     correspond_struct_idx[
                                                                                                         i],
                                                                                                     correspond_struct_idx[
                                                                                                         i + 1],
                                                                                                     structure,
                                                                                                     next_structure,
                                                                                                     colors, c1)

    # 集束调整
    structure = bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure)
    i = 0
    # 由于经过bundle_adjustment的structure，会产生一些空的点（实际代表的意思是已被删除）
    # 这里删除那些为空的点
    while i < len(structure):
        if math.isnan(structure[i][0]):
            structure = np.delete(structure, i, 0)
            colors = np.delete(colors, i, 0)
            i -= 1
        i += 1

    np.save('./structure.npy', structure)
    np.save('./colors.npy', colors[:,[2,1,0]])#从bgr转换到rgb

def PathPlan():
    tripoints3d = np.load('./structure.npy')
    colors = np.load('./colors.npy')

# 保存重建结果
    save_path = './original.ply'
    create_output(tripoints3d, colors, save_path)

    filtered_path = "./filtered_mesh.ply"
    rot_path = './rot.ply'
    result_path = './mesh_with_path.ply'

# 点云滤波

    filtered, colors = meshFilter(tripoints3d, colors)
    create_output(filtered, colors, filtered_path)  # 滤波后的三维点云文件

# 将点云中地面旋转至于xoy平面平行
    # p1 p2 p3是三个手动选择的同一平面上的点
    p1 = np.array([1.59183, 1.59993, 6.10826])
    p2 = np.array([1.05749, 0.51910, 7.84329])
    p3 = np.array([-1.2133, 0.281396, 7.83612])

    rot = rotatePC(filtered, p1, p2, p3)
    create_output(rot, colors, rot_path)  # 滤波+旋转后的三维点云文件

# 将旋转后的图转换成障碍图
    t = transform(15, 50, 100)# 半径10， 点云坐标放大100倍，边缘留出50个像素空隙
    obstacle = t.fillMesh2D(rot[:,:-1])
    # cv2.imshow("original obstacle", obstacle)
    # cv2.waitKey(0)

# 路径规划
    # 确定起点，终点
    end = (625, 200)
    start = (30, 280)

    # 求路径
    kernel = cv2.getStructuringElement(ksize=(3, 3), shape=cv2.MORPH_RECT)
    obstacle = cv2.dilate(obstacle, kernel)
    plt.imshow(obstacle)
    plt.show()
    path = get_path(obstacle, start, end)
    for r, c in path:
        obstacle[r][c] = 10

    plt.imshow(obstacle)
    plt.show()

    path = np.array(path)
    path = np.stack((path[:, 1], path[:, 0])).T

# 恢复路径的三维空间坐标
    z = np.min(rot[:, 2])
    path3D = t.recover3D(path, z)

    # 结果可视化
    path_color = np.zeros((path3D.shape[0], 3))
    mesh_with_path = np.vstack((rot, path3D))
    color_all = np.vstack((colors, path_color))
    create_output(mesh_with_path, color_all, result_path)

if __name__ == '__main__':
    #SFM()

    #SFM结果已经保存，可直接运行路径规划算法查看可视化结果
    PathPlan()
