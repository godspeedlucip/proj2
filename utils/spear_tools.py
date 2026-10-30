import cv2
import csv
import glob
import numpy as np
import xml.dom.minidom
from skimage.restoration import denoise_tv_chambolle as TV
import os
import pydicom
import pydicom.uid
from scipy.ndimage import gaussian_filter
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def denoiseTV(img, delta):
    for i in range(img.shape[0]):  # shape[0]读取矩阵第一维度的长度，即数组的行数
        minv = img[i].min()
        maxv = img[i].max()
        img[i] = TV((img[i] - minv) / (maxv - minv), weight=delta, n_iter_max=100) * (maxv - minv) + minv
    return img

def readCSVBin(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        column = [np.int8(row[0]) for row in reader]  # 提取csv文件中的第一列
    return np.array(column)

def readXMLAngle(fileName):
    dom = xml.dom.minidom.parse(fileName)  # 读取xml文件
    cc = dom.getElementsByTagName('GantryAngle')  # 获取“龙门架角度”这一标签下的数据
    angles = np.zeros([len(cc)], dtype=np.float32)  # 创建“龙门架角度”标签数量的列表
    for i in range(len(cc)):
        angles[i] = np.float32(cc[i].firstChild.data)
    return angles

def readProj(dirName, w, h):

    projFileList = glob.glob(dirName + '*.bin')  # 获取指定目录下的所有.bin文件
    w, h = w, h
    proj = np.zeros([len(projFileList), w, h], np.float32)  # 设置一个空的以（.bin文件的数量，weight宽度，height高度）作为长宽高的三维矩阵

    for projIndex in range(proj.shape[0]):  # shape[0]读取矩阵第一维度的长度，即数组的行数
        p = np.reshape(np.fromfile(dirName + 'Proj_%05d.bin' % (projIndex + 1), np.float32),
                       [w, h])  # 将bin里面的数据变成w*h大小的矩阵
        proj[projIndex] = p  # cv2.GaussianBlur(p, (3,3), 2.5)  按顺序将.bin文件对应的矩阵填入对应位置
    return proj  # 返回每个.bin文件及其数据组成的数组

def weightGen(w, delta):
    weight = np.zeros([1, delta * 2])
    for i in range(delta * 2):
        mu = (i + 1) / (2 * delta)
        weight[0, i] = 10 * mu ** 3 - 15 * mu ** 4 + 6 * mu ** 5  # 10* mu^3 - 15* mu^4 + 6* mu^5
    return np.tile(weight, [w, 1])

def padding_linatech(proj, len):
    w, h, z = proj.shape
    projPadd = np.zeros((w, h+len*2, z), dtype=np.float32)
    projPadd[:, len:len + h, :] = proj
    projPadd[:, :len, :] = proj[:, len - 1::-1, :]
    projPadd[:, len + h:, :] = proj[:, -len:, :]
    return projPadd

def maskGenerate_linatech(H, W, Z, R):
    FOV = np.zeros((H, W, Z), dtype=np.float32)
    for i in range(Z):
        FOV[:,:,i] = circleCompute(R, H, W)
    return FOV

def circleCompute(r, H, W):
    mask = np.zeros((H, W))
    # 计算圆心位置
    center_x, center_y = (W - 1) / 2, (H - 1) / 2
    # 生成网格坐标
    Y, X = np.ogrid[:H, :W]

    # 计算距离并创建掩模
    distance = (X - center_x) ** 2 + (Y - center_y) ** 2
    mask[distance <= r ** 2] = 1

    return mask

def readProj_linatech(rootdir, num, I0, len):
    '''
    用来读取雷泰原始的投影数据，文件格式是dcm
    :param rootdir: 原始投影文件目录
    :param num: 投影文件数量
    :param I0: 投影背景
    :param len: padding
    :return: proj 投影数据， angles 投影角度
    '''
    angles = np.zeros((1, 359), dtype=np.single)

    data = np.zeros((1408, 1408, num), dtype=np.single)

    for i in range(1, num+1):
        filename = '{}{:03d}.dcm'.format(rootdir, i*2)
        dicom_file = pydicom.dcmread(filename, force=True)
        angle = dicom_file.get('GantryAngle', None)
        angles[:, i-1] = angle
        dicom_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        Image = dicom_file.pixel_array
        # print(Image.shape)
        # Image = np.transpose(Image, (2, 0, 1))
        Image = -np.log(Image / (I0 + 1e-7))
        Image[np.isinf(Image)] = 0

        # 定义多项式系数
        a = 3.99567731237236
        b = -25.4493077640396
        c = 55.8284068254555
        d = 11.9551036611716
        e = 0.475674868518792

        Image = a * np.power(Image, 4) + b * np.power(Image, 3) + c * np.power(Image, 2) + d * Image + e

        data[:, :, i-1] = gaussian_filter(Image, sigma=1)

    proj = np.zeros((352, 704, num), dtype=np.single)
    for i in range(0, 2):
        for j in range(0, 4):
            proj += data[j::4, i::2, :]
    proj /= 8
    # proj = np.transpose(proj,(2, 0, 1))
    # print(proj.shape)
    angles = angles.flatten()
    return proj, angles


# def projPadding(proj, length=100):
#     z, w, h = proj.shape
#     CProj = np.zeros([z, w, h + length * 2])
#     CProj[:, :, length:-length] = proj
#     for i in range(length):
#         CProj[:, :, i] = np.sin(i * np.pi / (2 * (length - 1))) * (2 * proj[:, :, 0] - proj[:, :, length - i])
#         CProj[:, :, length + h + i] = np.cos(i * np.pi / (2 * (length - 1))) * (2 * proj[:, :, -1] - proj[:, :, -i])
#     CProj[CProj < 0] = 0
#     return CProj