#!/usr/bin/env python
# coding: utf-8
import cv2
from PIL import ImageEnhance
import numpy as np
import urllib.request
from sys import argv
import base64
import redis
import socket
#import socketserver

def cv_imread(file_path):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def cv_imwrite(save_path,for_save_image):
    cv2.imencode('.jpg',for_save_image)[1].tofile(save_path)

def base64_to_image(base64_img):
    img_data = base64.b64decode(base64_img)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img

def image_to_base64(image_array):
    image = cv2.imencode('.jpg',image_array)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def image_to_bytes(image):
    img_encode = cv2.imencode('.jpg', image)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    return str_encode

"""
图片裁切
cut_range = x1,x2,y1,y2
x1,x2,y1,y2:裁切图片x1-x2行，y1-y2列
"""
def cut(image_path,cut_range):
    cut_range = cut_range.split(",")
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    cut_image = image[int(cut_range[0]):int(cut_range[1]),int(cut_range[2]):int(cut_range[3])]
    return image_to_base64(cut_image)

"""
图像缩放
k = k1,k2
k1:高度缩放倍数
k2:宽度缩放倍数
"""
def resize(image_path,k):
    k = k.split(",")
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    h,w = image.shape[:2]
    size = (int(w*float(k[0])),int(h*float(k[1])))
    resize_image = cv2.resize(image,size,interpolation = cv2.INTER_LANCZOS4)
    return image_to_base64(resize_image)


"""
图片翻转
n:翻转方向
    水平翻转:n=1
    垂直翻转:n=0
    水平垂直翻转:n=-1
"""
def flip(image_path,n):
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    flip_image = cv2.flip(image,int(n))
    return image_to_base64(flip_image)

"""
图片旋转
angel:旋转角度
"""
def rotate(image_path,angle):
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    M = cv2.getRotationMatrix2D((cx,cy),-int(angle),1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    return image_to_base64(cv2.warpAffine(image,M,(nW,nH)))

"""
图片灰度化
n:灰度化处理方法
    n=1：加权平均灰度处理方法
    n=2：平均灰度处理方法
    n=3：最大值灰度处理方法
"""
def gray(image_path,n):
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    if int(n) == 0:
        return image_to_base64(image)
    else:
        if len(image.shape) == 2:
            gray_image = image
        elif int(n) == 1:
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        elif int(n) == 2:
            B,G,R = cv2.split(image)
            gray_pix = B/3+G/3+R/3
            gray_image = np.uint8(gray_pix)
        else:
            B,G,R = cv2.split(image)
            max_BG = cv2.max(B,G)
            max_pix = cv2.max(max_BG,R)
            gray_image = np.uint8(max_pix)
        return image_to_base64(gray_image)

"""
调整图片亮度、对比度
alpha:增益（alpha>0）
beta:偏置
对比度调整:同时改变alpha、beta的值
亮度调整:单独改变beta的值
"""
def contrast_brightness(image_path,setting):
    alpha = float(setting.split(",")[0])
    beta = int(setting.split(",")[1])
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    res = np.uint8(np.clip((alpha * image + beta), 0, 255))
    return image_to_base64(res)

"""
图片锐化
kernel_sharpen:锐化卷积核
"""
def sharpen(image_path,k):
    k = int(k)
    if k == 0:
        kernel_sharpen = np.array([
            [ 0, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 0]
        ])
    elif k == 1:
        kernel_sharpen = np.array([
            [ 0,-1, 0],
            [-1, 5,-1],
            [ 0,-1, 0]
        ])
    elif k == 2:
        kernel_sharpen = np.array([
            [-1,-1,-1],
            [-1, 9,-1],
            [-1,-1,-1]
        ])
    else:
        k = k-1
        kernel_sharpen = np.array([
            [-k, -k  ,-k],
            [-k,8*k+1,-k],
            [-k, -k  ,-k]
        ])
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    sharpen_image = cv2.filter2D(image,-1,kernel_sharpen)
    return image_to_base64(sharpen_image)

"""
图片饱和度
increment:饱和度增量，范围-1到1
"""
def saturation(image_path,increment):
    increment = float(increment)
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    enh_col = ImageEnhance.Color(image)
    image_out = enh_col.enhance(increment)
    return image_to_base64(image_out)

"""
边缘检测
canny边缘检测
sable边缘检测
"""
def edge(image_path,k):
    # image=cv_imread(image_path)
    image = base64_to_image(image_path)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if int(k) == 1:
        img=cv2.GaussianBlur(gray,(3,3),0)
        dst=cv2.Canny(img,50,50)
        return image_to_base64(dst)
		#return dst
    else:
        imgInfo=gray.shape
        height=imgInfo[0]
        weight=imgInfo[1]
        dst=np.zeros((height,weight,1),np.uint8)
        for i in range(0,height-2):
            for j in range(0,weight-2):
                gy=gray[i,j]*1+gray[i,j+1]*2+gray[i,j+2]*1-gray[i+2,j]-2*gray[i+2,j+1]-gray[i+2,j+2]*1
                gx=gray[i,j]*1-gray[i,j+2]+gray[i+1,j]*2-2*gray[i+1,j+2]+gray[i+2,j]-gray[i+2,j+2]
                grad=math.sqrt(gx*gx+gy*gy)
                if grad>50:
                    dst[i,j]=255
                else:
                    dst[i,j]=0
        return image_to_base64(dst)

"""
图像去噪
滤波种类：均值滤波,k=1
         高斯滤波,k=2
         中值滤波,k=3
         双边滤波,k=4
         高通滤波,k=5
         低通滤波,k=6
"""
def denoising(image_path,k):
    def low_pass_filter(img_in):
        dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(
            cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        rows, cols = img_in.shape
        crow, ccol = rows / 2, cols / 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[int(crow - 10):int(crow + 10), int(ccol - 10):int(ccol + 10)] = 1
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        img_out = img_back
        return img_out
    def high_pass_filter(img_in):
        dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(
            cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        rows, cols = img_in.shape
        crow, ccol = rows / 2, cols / 2
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[int(crow - 10):int(crow + 10), int(ccol - 10):int(ccol + 10)] = 0
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        img_out = img_back
        return img_out
    
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    if int(k) == 1:
        image_out = cv2.blur(image, (5,5))
    elif int(k) == 2:
        image_out = cv2.GaussianBlur(image,(5,5),0)
    elif int(k) == 3:
        image_out = cv2.medianBlur(image, 5)
    elif int(k) == 4:
        image_out = cv2.bilateralFilter(image,9,75,75)
    elif int(k) == 5:
        image_out = low_pass_filter(image)
    else:
        image_out = high_pass_filter(image)
    return image_to_base64(image_out)

"""
傅里叶变换
k_n = k,n
k=1 高通滤波
k=2 低通滤波
n 频率
"""
def fft(image_path,k_n):
    k = int(k_n.split(",")[0])
    n = int(k_n.split(",")[1])
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    if len(image.shape) == 3:
        img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    else:
        img = image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2
    fshift[crow-30:crow+30, ccol-30:ccol+30] = k
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    image_out = np.abs(img_back)
    return image_to_base64(image_out)

"""
边界提取
k = 1 自适应
k = B,G,R
"""
def bound(image_path,k):
    def histing(x):
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(band[x])
        hist = cv2.calcHist([band[x]], [0], None, [256], [minVal, maxVal])
        n = hist.tolist().index(max(hist))
        return n
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    fsrc = np.array(image, dtype=np.float32) /255.0
    band = cv2.split(fsrc)
    try:
        k = int(k)
    except ValueError as e:
        k = k.split(",")
    if k == 1:
        B_G_R = np.array([histing(0),histing(1),histing(2)])
    else:
        # k = k.split(",")
        B_G_R = np.array([int(k[0]),int(k[1]),int(k[2])])
    upper = B_G_R + 20
    lower = B_G_R - 20
    mask = cv2.inRange(image,lower,upper)
    (_,contours,hicrarchy) = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    all_Image = image.copy()
    cv2.drawContours(all_Image,contours,-1,(0,0,255),2)
    bound_img = image.copy()
    contours.sort(key=len,reverse=True)
    cv2.drawContours(bound_img,[contours[0]],-1,(0,0,255),2)
    return image_to_base64(bound_img)

"""
直方图增强
k=1，线性变换增强
k=2，直方图正规化
k=3，直方图均衡化
"""
def hist_enhance(image_path,k):
    def linear_transform(img, a, b):
        out = a * img + b
        out[out > 255] = 255
        out = np.around(out)
        out = out.astype(np.uint8)
        return out
    def normalize_transform(gray_img):
        Imin, Imax = cv2.minMaxLoc(gray_img)[:2]
        Omin, Omax = 0, 255
        a = float(Omax - Omin) / (Imax - Imin)
        b = Omin - a * Imin
        out = a * gray_img + b
        out = out.astype(np.uint8)
        return out
    def equalize_transfrom(gray_img):
        return cv2.equalizeHist(gray_img)
    
    # img = cv_imread(image_path)
    image = base64_to_image(image_path)
    if int(k) == 1:
        # a = 2, b=10
        linear_img = linear_transform(img, 2.0, 10)
        return image_to_base64(linear_img)
		#return linear_img
    elif int(k) == 2:
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        b_out = normalize_transform(b)
        g_out = normalize_transform(g)
        r_out = normalize_transform(r)
        nor_out = np.stack((b_out, g_out, r_out), axis=-1)
        return image_to_base64(nor_out)
		#return nor_out
    else:
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        b_out = equalize_transfrom(b)
        g_out = equalize_transfrom(g)
        r_out = equalize_transfrom(r)
        equa_out = np.stack((b_out, g_out, r_out), axis=-1)
        return image_to_base64(equa_out)

"""
图像去雾
k为滤波核的大小,奇数,默认值为15
"""
def dehaze(image_path,k):
    def dark_channel(img, size = 15):
        r, g, b = cv2.split(img)
        min_img = cv2.min(r, cv2.min(g, b))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dc_img = cv2.erode(min_img,kernel)
        return dc_img

    def get_atmo(img, percent = 0.001):
        mean_perpix = np.mean(img, axis = 2).reshape(-1)
        mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
        return np.mean(mean_topper)

    def get_trans(img, atom, n, w = 0.95):
        x = img / atom
        t = 1 - w * dark_channel(x, n)
        return t

    def guided_filter(p, i, r, e):
        mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
        corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + e)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
        q = mean_a * i + mean_b
        return q

    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    img = image.astype('float64') / 255
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    atom = get_atmo(img)
    trans = get_trans(img, atom, int(k))
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)

    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
    image_out = result*255
    return image_to_base64(image_out)

"""
道路提取
"""
def road_extraction(image_path,k):
    def sharpen(img):
	    im=img
	    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	    im = cv2.filter2D(im, -1, kernel)
	    return im
    def auto_canny(image, sigma=0.33):
	    v = np.median(image)
	    lower = int(max(0, (1.0 - sigma) * v))
	    upper = int(min(255, (1.0 + sigma) * v))
	    edged = cv2.Canny(image, lower, upper)
	    return edged
    def blobdetect(img):
	    im = img
	    inp=img
	    ret,thresh = cv2.threshold(im,100,255,0)
	    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	    img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
	    mask = np.ones(im.shape[:2], dtype="uint8") * 255
	    for c in contours:
		    if cv2.contourArea(c)<150000:
			    cv2.drawContours(mask, [c], -1, 0, -1)
	    img2= cv2.bitwise_and(img, img, mask=mask)
	    return img2
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    sharped=sharpen(image)
    frame=cv2.medianBlur(image,5)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Z = frame.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 20
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    resd = center[label.flatten()]
    resd2 = resd.reshape((frame.shape))
    cluster=resd2
    resd2=cv2.cvtColor(resd2,cv2.COLOR_BGR2GRAY)
    sharpcluster=sharpen(resd2)
    edges = cv2.Canny(frame,400,600)
    edges3=cv2.Canny(resd,200,260)
    edgesharped=auto_canny(sharpcluster)
    smoothmeans=cv2.GaussianBlur(resd2,(5,5),0)
    equ = cv2.equalizeHist(resd2)
    final=blobdetect(equ)
    retval, threshold = cv2.threshold(final, 220, 255, cv2.THRESH_BINARY)
    image_out = threshold
    return image_to_base64(image_out)

"""
图像拼接
"""
def stitch(image_path,k):
    def get_sift_homography(img1, img2):
	    sift = cv2.xfeatures2d.SIFT_create()
	    k1, d1 = sift.detectAndCompute(img1, None)
	    k2, d2 = sift.detectAndCompute(img2, None)
	    bf = cv2.BFMatcher()
	    matches = bf.knnMatch(d1,d2, k=2)
	    verify_ratio = 0.8
	    verified_matches = []
	    for m1,m2 in matches:
		    if m1.distance < 0.8 * m2.distance:
			    verified_matches.append(m1)
	    min_matches = 8
	    if len(verified_matches) > min_matches:
		    img1_pts = []
		    img2_pts = []
		    for match in verified_matches:
			    img1_pts.append(k1[match.queryIdx].pt)
			    img2_pts.append(k2[match.trainIdx].pt)
		    img1_pts = np.float32(img1_pts).reshape(-1,1,2)
		    img2_pts = np.float32(img2_pts).reshape(-1,1,2)
		    M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
		    return M
	    else:
		    print('Error: Not enough matches')
    def get_stitched_image(img1, img2, M):
	    w1,h1 = img1.shape[:2]
	    w2,h2 = img2.shape[:2]
	    img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
	    img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)
	    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)
	    result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)
	    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
	    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
	    transform_dist = [-x_min,-y_min]
	    transform_array = np.array([[1, 0, transform_dist[0]], 
								    [0, 1, transform_dist[1]], 
								    [0,0,1]]) 
	    result_img = cv2.warpPerspective(img2, transform_array.dot(M), 
									    (x_max-x_min, y_max-y_min))
	    result_img[transform_dist[1]:w1+transform_dist[1], 
				    transform_dist[0]:h1+transform_dist[0]] = img1
	    return result_img
    path1 = image_path.split(",")[0]
    path2 = image_path.split(",")[1]
    # image1 = cv_imread(path1)
    # image2 = cv_imread(path2)
    image1 = base64_to_image(path1)
    image2 = base64_to_image(path2)
    M = get_sift_homography(image1,image2)
    result_image = get_stitched_image(image2,image1,M)
    return image_to_base64(result_image)

"""
低照度增强
"""
def dark_enhance(image_path,w):
    def zmMinFilterGray(src, r=7):
        if r <= 0:
            return src
        h, w = src.shape[:2]
        I = src
        res = np.minimum(I, I[[0] + [x for x in range(h-1)], :])
        res = np.minimum(res, I[[x for x in range(1, h)] + [h - 1], :])
        I = res
        res = np.minimum(I, I[:, [0] + [x for x in range(w - 1)]])
        res = np.minimum(res, I[:, [x for x in range(1, w)] + [w - 1]])
        return zmMinFilterGray(res, r - 1)
    def guidedfilter(I, p, r, eps):
        height, width = I.shape
        m_I = cv2.boxFilter(I, -1, (r, r))
        m_p = cv2.boxFilter(p, -1, (r, r))
        m_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = m_Ip - m_I * m_p
        m_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = m_II - m_I * m_I
        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I
        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        return m_a * I + m_b
    def getV1(m, r, eps, w, maxV1):
        V1 = np.min(m, 2)
        V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)
        bins = 2000
        ht = np.histogram(V1, bins)
        d = np.cumsum(ht[0]) / float(V1.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
        V1 = np.minimum(V1 * w, maxV1)
        return V1, A
    def dh(m, w, r=81, eps=0.001, maxV1=0.80, bGamma=False):
        Y = np.zeros(m.shape)
        V1, A = getV1(m, r, eps, w, maxV1)
        for k in range(3):
            Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)
        Y = np.clip(Y, 0, 1)
        if bGamma:
            Y = Y ** (np.log(0.5) / np.log(Y.mean()))
        return Y
    def image_convet(img):
        img[:, :, :] = 255 - img[:, :, :]
        return img

    # img = cv_imread(image_path)
    img = base64_to_image(image_path)
    # img = cv2.resize(img, (640, 480))
    img = image_convet(img)
    m = dh( img / 255.0,float(w)) * 255
    light_image = image_convet(m)
    return image_to_base64(light_image)

"""
直方图匹配
"""
def hist_match(image_path,k):
    def calc_pdf_cdf ( img ) :
	    height = img.shape[0]
	    width  = img.shape[1]
	    pdf = cv2.calcHist([img],[0],None,[256],[0,256] )
	    pdf /= (width*height)
	    pdf = np.array( pdf )
	    cdf = np.zeros( 256 )
	    cdf[0] = pdf[0]
	    for i in range(1,256) :
		    cdf[i] = cdf[i-1] + pdf[i]
	    return pdf, cdf

    srcimg_name = image_path.split(",")[0]
    refimg_name = image_path.split(",")[1]

    # src_img = cv2.cvtColor( cv_imread( srcimg_name ), cv2.COLOR_RGB2GRAY)
    # ref_img = cv2.cvtColor( cv_imread( refimg_name ), cv2.COLOR_RGB2GRAY)
    src_img = cv2.cvtColor(base64_to_image(srcimg_name), cv2.COLOR_RGB2GRAY)
    ref_img = cv2.cvtColor(base64_to_image(refimg_name), cv2.COLOR_RGB2GRAY)

    src_pdf, src_cdf = calc_pdf_cdf(src_img)
    ref_pdf, ref_cdf = calc_pdf_cdf(ref_img)

    mapping = np.zeros(256, dtype = int)

    for i in range(256) :
	    #search j such that src_cdf(i) = ref_cdf(j)
	    # and set mapping[i] = j
	    for j in range(256) :
		    if ref_cdf[j] >= src_cdf[i] :
			    break
	    mapping[i] = j

    #gen output image
    out_img = np.zeros_like(src_img, dtype = np.uint8)
    for i in range(256) :
	    out_img[ src_img == i ] = mapping[i]
    # cv2.imwrite( outimg_name, out_img)
    return image_to_base64(out_img)

"""
颜色转换
"""
def convert_color(image_path,k):
    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    if int(k) == 1:
        out_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    elif int(k) == 2:
        out_image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    elif int(k) == 3:
        out_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    else:
        out_image = cv2.cvtColor(image,cv2.COLOR_BGR2LUV)
    return image_to_base64(out_image)

"""
图像融合
"""
def fusion(image_path,k):
    path1 = image_path.split(",")[0]
    path2 = image_path.split(",")[1]
    image1 = base64_to_image(path1)
    image2 = base64_to_image(path2)
    if int(k) == 0:
        image_out = cv2.addWeighted(src1,0.5,src2,0.5,0)
        return image_to_base64(image_out)
    else:
        imageSize = image1.size
        allImage = np.concatenate((image1.reshape(1, imageSize), image2.reshape(1, imageSize)), axis=0)
        covImage = np.cov(allImage)
        D, V = np.linalg.eig(covImage)
        if D[0] > D[1]:
            a = V[:,0] / V[:,0].sum()
        else:
            a = V[:,1] / V[:,1].sum()
        image_out = image1*a[0] + image2*a[1]
        return image_to_base64(image_out)

"""
图像校正
"""
def correction(image_path,k):
    import imutils
    def order_points(pts):
        rect = np.zeros((4,2), dtype = "float32")
        s = np.sum(pts, axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
        widthB = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
        heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0,0],
            [maxWidth - 1, 0],
            [maxWidth -1, maxHeight -1],
            [0, maxHeight -1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))    
        return warped
    def preProcess(image):
        ratio = image.shape[0] / 500.0
        image = imutils.resize(image, height=500)
        grayImage  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
        edgedImage = cv2.Canny(gaussImage, 75, 200)
        cnts = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in cnts:
            peri = cv2.arcLength(c, True)  # Calculating contour circumference
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
        return  screenCnt, ratio

    # image = cv_imread(image_path)
    image = base64_to_image(image_path)
    screenCnt, ratio = preProcess(image)
    out_image = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)
    return image_to_base64(out_image)

Fun_map = {'cut':cut,'resize':resize,'flip':flip,'rotate':rotate,
            'gray':gray,'contrast_brightness':contrast_brightness,
            'sharpen':sharpen,'saturation':saturation,'denoising':denoising,
            'bound':bound,'fft':fft,'edge':edge,'hist_enhance':hist_enhance,
            'dehaze':dehaze,'road_extraction':road_extraction,'stitch':stitch,
            'dark_enhance':dark_enhance,'hist_match':hist_match,'convert_color':convert_color,
            'fusion':fusion,'correction':correction}

# class myTCPhandler(socketserver.BaseRequestHandler):
#     def handle(self):
#         while True:
#             try:
#                 self.data = self.request.recv(1024).decode('UTF-8', 'ignore').strip()
#                 if not self.data : break
#                 print("\n" + "接收到参数" + self.data)
#                 key = self.data.split(",")[0]
#                 input_base64=r.get(key)
#                 function = self.data.split(",")[1]
#                 parameter = self.data.split(",")[2]
#                 output_image = Fun_map[function](input_base64,parameter)
#                 self.feedback_data =(output_image).encode("utf8")
#                 print("key_value设置成功")
#                 r.set(key,self.feedback_data)
#                 a = "success"
#                 self.request.sendall(a.encode("utf8"))
#             except:
#                 print("连接断开" + "\n")
#                 break
            
# pool = redis.ConnectionPool(host='192.168.1.253', port=6379, decode_responses=True)
# r = redis.Redis(connection_pool=pool)
# print('sever on')

# host = '192.168.1.253'
# port = 9007
# server = socketserver.ThreadingTCPServer((host,port),myTCPhandler)
# server.serve_forever()

pool = redis.ConnectionPool(host="192.168.1.253", port=6379, decode_responses=True)
r = redis.Redis(connection_pool=pool)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("192.168.1.253", 9007))
sock.listen(1)

def process(connection,address):
    try:
        recive_info = connection.recv(1024)
        print("\n" + "接收到参数" + recive_info)
        key = recive_info("#")[0]
        input_base64=r.get(key)
        function = recive_info.split("#")[1]
        parameter = recive_info.split("#")[2]
        output_image = Fun_map[function](input_base64,parameter)
        feedback_data =(output_image).encode("utf8")
        print("key_value设置成功")
        r.set(key,feedback_data)
        a = "success"
        connection.send(a.encode("utf8"))
    except Exception as e:
            print(e)
    print("closing one connection")
    connection.close()


while True:
    connection,address = sock.accept()
    t = threading.Thread(target=process, args=(connection,address))
    t.start()
    # try:
    #     recive_info = connection.recv(1024)
    #     print("\n" + "接收到参数" + recive_info)
    #     key = recive_info("#")[0]
    #     input_base64=r.get(key)
    #     function = recive_info.split("#")[1]
    #     parameter = recive_info.split("#")[2]
    #     output_image = Fun_map[function](input_base64,parameter)
    #     feedback_data =(output_image).encode("utf8")
    #     print("key_value设置成功")
    #     r.set(key,feedback_data)
    #     a = "success"
    #     connection.send(a.encode("utf8"))
    # except Exception as e:
    #         print(e)
    # print("closing one connection")
    # connection.close()