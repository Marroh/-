import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 输入点云坐标和颜色，保存为方便可视化的ply文件
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

# 输入点云（pc），以及地面上的三个点坐标，用来计算地面的法向量
def rotatePC(pc, p1, p2, p3):
    norm_vector = np.cross((p1-p2),(p1-p3))
    #print(norm_vector)
    z = np.array([0, 0, 1])# 旋转至z轴
    k = np.cross(norm_vector, z)
    k = k / math.sqrt(k.dot(k.T))

    # 计算旋转矩阵的辅助矩阵
    K=np.array([[0,-k[2],k[1]],
                [k[2],0,-k[0]],
                [-k[1],k[0],0]])

    theta = math.acos(z.dot(norm_vector.T)/math.sqrt(z.dot(z.T)*norm_vector.dot(norm_vector.T)))
    print('angle', theta*180/math.pi)

    # 计算旋转矩阵
    R = np.eye(3) + K * math.sin(theta) + (1-math.cos(theta))*K.dot(K)
    rot_a = pc.dot(R.T)

    return rot_a

# 点云滤波
def meshFilter(mesh, color):
    filter = DBSCAN(eps=0.30,  # 邻域半径
    min_samples=10,    # 最小样本点数，MinPts
    metric='euclidean',
    metric_params=None,
    algorithm='auto', # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
    leaf_size=30, # balltree,cdtree的参数
    p=None, #
    n_jobs=1)

    class_pred = filter.fit_predict(mesh)
    dele = []
    for i, c in enumerate(class_pred):
        if c == -1:
            dele.append(i)

    # 删除不是核心目标的点对应的数据
    mesh = np.delete(mesh, dele, axis=0)
    color = np.delete(color, dele, axis=0)

    return mesh, color

# 这个类用于将点云的投影填充成障碍图，以及把障碍图中路径变换到三维空间
class transform():
    def __init__(self, radius, margin, scale):
        self.radius = radius
        self.margin = margin
        self.scale = scale# 根据SFM具体尺度进行放缩，

        #初始化
        self.x_min = 0
        self.y_min = 0
        self.shape=[0,0]

    def recover3D(self, pathArr, z):
        # 先恢复x，y坐标
        pathArr = np.stack((pathArr[:, 0], self.shape[0]-pathArr[:, 1])).T
        print('path shape',pathArr.shape)
        path2D = (pathArr-self.margin) / self.scale + [self.x_min,self.y_min]
        path3D = np.hstack((path2D, np.ones((path2D.shape[0],1))*z))
        return path3D

    #shape就填ymax和xmax
    def fillMesh2D(self, mesh):
        # 将点云对齐到原点
        self.x_min = np.min(mesh[:,0])
        self.y_min = np.min(mesh[:,1])

        mesh = (mesh - [self.x_min,self.y_min])*self.scale + self.margin#点云坐标放大到合适的尺度，四周都空出margin个单位长度的距离
        mesh = mesh.astype(np.int)

        x_max = np.max(mesh[:,0])
        y_max = np.max(mesh[:,1])

        # 投影区域周围空出margin大小间隙
        self.shape = [y_max+self.margin,x_max+self.margin]
        map = np.zeros(self.shape)

        # 每个点对应一个障碍区域
        for point in mesh:
            x = point[0]
            y = self.shape[0] - point[1]
            # left
            if x-self.radius<0: left = x
            else: left = self.radius
            # right
            if x+self.radius>=self.shape[1]: right = self.shape[1]-x-1
            else: right = self.radius
            # up
            if y-self.radius<0: up = y
            else: up = self.radius
            # down
            if y+self.radius>=self.shape[0]: down = self.shape[0]-y-1
            else: down = self.radius

            # 以每个点云投影为中心，画一个矩形作为障碍物
            map[y-up:y+down, x-left:x+right] = 1
        return map