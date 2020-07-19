import numpy as np
from pathlib import Path, PurePath
from skimage import io
from skimage import feature, measure, transform, morphology, color, filters
from scipy import ndimage
import math
import sys
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks

class photosDict:
    def __init__(self, path, count):
        self.dict = [];
        self.loadPhotos(path, count)

    def loadPhotos(self, path, count):
        for i in range(1, count+1):
            filePath = Path(path + "/img_" + str(i) + ".jpg")
            print("Loading file: "+str(filePath))
            image = io.imread(filePath)
            self.dict.append(image)

def anglesFromContours(contour):
    ret = np.zeros(len(contour)-1)
    for i in range(len(contour)-1):
        a = 0
        b = 0
        if i==0:
            a = math.atan2(contour[i + 1, 0] - contour[i, 0], contour[i + 1, 1] - contour[i, 1])
            b = math.atan2(contour[i - 2, 0] - contour[i, 0], contour[i - 2, 1] - contour[i, 1])
        else:
            a = math.atan2(contour[i + 1, 0] - contour[i, 0], contour[i + 1, 1] - contour[i, 1])
            b = math.atan2(contour[i - 1, 0] - contour[i, 0], contour[i - 1, 1] - contour[i, 1])
        if a<0:
            a = 2 * math.pi - abs(a)
        if b<0:
            b = 2 * math.pi - abs(b)

        ret[i] = a-b
    return ret

def makeMargin(image, margin):
    for i in range(image.shape[1]):
        for ii in range(image.shape[0]):
            if (i<margin) or (ii<margin) or (i>image.shape[1]-margin-1) or (ii>image.shape[0]-margin-1):
                image[ii, i] = 0
    return image


def crossingPoints(image, angles, distances):
    x = []
    y = []

    for i in range(len(angles)):
        for ii in range(i+1, len(angles)):
            ctgT1 = np.tan(np.pi/2-angles[i])
            ctgT2 = np.tan(np.pi/2-angles[ii])
            sinT1 = np.sin(angles[i])
            sinT2 = np.sin(angles[ii])
            d1 = distances[i]
            d2 = distances[ii]
            x.append((d2/sinT2-d1/sinT1)/(ctgT1-ctgT2))
            y.append(ctgT1*x[len(x)-1]+d1/sinT1)

    for i in range(len(x)-1, -1, -1):
        if (x[i] >= image.shape[1]) or (y[i] >= image.shape[0]) or (x[i] < 0) or (y[i] < 0):
            """x.pop(i)
            y.pop(i)"""
    print(image.shape)
    print(x)
    print(y)

def takeBiggestRegion(image):
    image = measure.label(image, connectivity=1)
    regionprops = measure.regionprops(image)

    maxArea = 0
    maxAreaIndex = 0
    for i in regionprops:
        if i.area > maxArea:
            maxArea = i.area
            maxAreaIndex = i.label

    image = image == maxAreaIndex

    return image

def takeRectangleContour(contour, image):
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    angles = anglesFromContours(contour)
    print(angles)
    angleDiff = []
    for i in angles:
        diff = abs((i+np.pi/2)-np.pi*round((i+np.pi/2)/np.pi))
        angleDiff.append(diff)
    contourQuarter = []
    for i in contour:
        if (i[0]<imageHeight/2) and (i[1]>imageWidth/2):
            contourQuarter.append(0)
        elif (i[0]>imageHeight/2) and (i[1]>imageWidth/2):
            contourQuarter.append(1)
        elif (i[0]>imageHeight/2) and (i[1]<imageWidth/2):
            contourQuarter.append(2)
        else:
            contourQuarter.append(3)

    newContour = np.zeros(10).reshape(5, 2)
    for i in range(4):
        points = []
        for ii in range(len(contour)-1):
            if contourQuarter[ii] == i:
                points.append(ii)

        bestAngleIndex = -1
        for ii in points:
            if (bestAngleIndex ==-1) or (angleDiff[bestAngleIndex] > angleDiff[ii]):
                bestAngleIndex = ii
        diff = np.pi/2;
        for ii in points:
            if bestAngleIndex == ii:
                continue
            else:
                if diff > (angleDiff[ii]-angleDiff[bestAngleIndex]):
                    diff = angleDiff[ii]-angleDiff[bestAngleIndex]
        if (1==len(points)) or (diff > np.pi/4):
            newContour[i] = contour[bestAngleIndex]
        else:
            newContour[i] = contour[bestAngleIndex]
        print(points)

    newContour[4] = newContour[0]
    print(newContour)
    return newContour



def detectPaper(image):
    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    grayscale = color.rgb2gray(image)
    ax[0].imshow(grayscale)

    block_size = 1555
    local_thresh = filters.threshold_local(grayscale, block_size)
    binary_local = grayscale > local_thresh
    ax[1].imshow(binary_local)

    binary_local = morphology.opening(binary_local, morphology.rectangle(5, 5))

    binary_local = morphology.closing(binary_local, morphology.rectangle(30, 30))

    binary_local = takeBiggestRegion(binary_local)

    binary_local = makeMargin(binary_local, 3)

    contours = measure.find_contours(binary_local, 0.5)
    contourPolygon = measure.approximate_polygon(contours[0], 50)
    paperContour = takeRectangleContour(contourPolygon, binary_local)
    print(contourPolygon)
    ax[2].imshow(binary_local)
    for n, contour in enumerate(paperContour):
        ax[2].plot(paperContour[:, 1], paperContour[:, 0], linewidth=2)



    plt.tight_layout()
    plt.show()

    """edges1 = feature.canny(grayscale, sigma=1)

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(edges1, theta=tested_angles)
    lines = hough_line_peaks(h, theta, d)
    print(lines)
    crossPoints = crossingPoints(grayscale, lines[1], lines[2])

    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    plt.show()"""

    """fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5),
                                        sharex=True, sharey=True)

    ax1.imshow(grayscale, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title(r'Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title(r'Canny filter, $\sigma=3$', fontsize=20)

    fig.tight_layout()

    plt.show()"""
    """plt.figure(figsize=(15, 10))
    io.imshow(i)
    plt.show()"""

if __name__ == "__main__":
    photos = photosDict(sys.argv[1], int(sys.argv[2]))

    for i in photos.dict:
        detectPaper(i)