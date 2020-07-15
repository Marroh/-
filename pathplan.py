import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(100000)
import cv2
# A*


def find_path(map:np.ndarray,start:tuple,end:tuple):
    R,C = map.shape
    start = np.array(start, dtype=np.int32)
    end = np.array(end, dtype=np.int32)
    move = np.array([[1,0],[-1,0],[0,1],[0,-1]])

    vis = np.zeros(shape=(R,C),dtype=np.int32)
    pre = np.zeros(shape=(R,C,2),dtype=np.int32)

    points = [start]

    while len(points) != 0:
        p = points.pop()
        r,c = p

        move_points = p + move
        inside_ind = inside(move_points,R,C)
        move_points = move_points[inside_ind]

        sorted_indices = rank(move_points,start,end,"c")
        move_points = move_points[sorted_indices]
        for mp in move_points:
            mr = mp[0]
            mc = mp[1]
            if vis[mr][mc] == 0 and map[mr][mc] == 0:
                points.append(mp)
                vis[mr][mc] = 1
                pre[mr][mc] = [r,c]

                if mr==end[0] and mc == end[1]:
                    return pre


def inside(P,R,C):

    index = (P[:, 0]>= 0)*(P[:, 1] >= 0)*(P[:, 0]<R)*(P[:, 1]<C)
    return index


def rank(P,start,end,mode):
    distance = np.square(P-end)
    '''
    if mode=="r":
        distance[:,0] = distance[:,0]*2
    else:
        distance[:,1] = distance[:,1]*2
    '''
    distance = np.sum(distance,axis=1,dtype=np.int32)
    sorted_indices = np.argsort(distance)
    return sorted_indices[::-1]


def print_path(pre,path,end,start):
    end_r,end_c = end
    start_r,start_c = start
    r = end_r
    c = end_c
    while r!=start_r or c!=start_c:
        pre_r,pre_c = pre[r][c]
        path.append(pre[r][c])
        r = pre_r
        c = pre_c


def get_path(map_o,start,end):

    pre = find_path(map_o,start,end)
    path = []
    print_path(pre,path,end,start)
    path.append([end[0],end[1]])

    return path



















