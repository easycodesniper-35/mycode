import numpy as np
import cv2

def rgb2hsi(image):
    rows = int(image.shape[0])
    cols = int(image.shape[1])
    b, g, r = cv2.split(image)

    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_img = image.copy()
    H, S, I = cv2.split(hsi_img)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            theta = float(np.arccos(num/den))

            if den == 0:
                    H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2*3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3*min_RGB/sum

            H = H/(2*3.14159265)
            I = sum/3.0
            
            hsi_img[i, j, 0] = H*255
            hsi_img[i, j, 1] = S*255
            hsi_img[i, j, 2] = I*255
    return hsi_img


if __name__ == "__main__":
    image_path = "1-MSS.tif"
    image = cv2.imread(image_path)
    hsi_image = rgb2hsi(image)
    cv2.imshow('test',hsi_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()