import numpy as np
from pathlib import Path, PurePath
from skimage import io, util
from skimage import feature, measure, transform, morphology, color, filters, draw
from scipy import ndimage
import math
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks
from scipy import ndimage as ndi
from skimage.filters import try_all_threshold
from skimage.transform import rescale, resize

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
import os
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from scipy import ndimage as ndi


class ModelNN(nn.Module):
    def __init__(self):
        super(ModelNN, self).__init__()
        self.lin1 = nn.Linear(784, 200)
        self.lin2 = nn.Linear(200, 80)
        self.lin3 = nn.Linear(80, 10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=1)
        return x


class PhotoDictPIRO:
    def __init__(self, input_path, count):
        self.dict = []
        self.load_photos(input_path, count)

    def load_photos(self, input_path, count):
        print_progress(0, count, prefix="Reading images:", suffix="Complete", bar_length=50)

        for i in range(0, count):
            # To be uncommented before running final tests
            # file_path = Path(input_path + "/" + str(i) + ".png")

            # To be commented/removed before running final tests
            file_path = Path(input_path + "/img_" + str(i + 1) + ".jpg")

            image = io.imread(file_path)
            self.dict.append(image)
            print_progress(i + 1, count, prefix="Reading images:", suffix="Complete", bar_length=50)


class PhotosDict:
    def __init__(self, path, count, first):
        self.dict = []
        self.load_photos(path, count, first)

    def load_photos(self, path, count, first):
        print_progress(0, count, prefix="Reading images:", suffix="Complete", bar_length=50)

        for i in range(first, first + count):
            file_path = Path(path + "/img_" + str(i) + ".jpg")
            image = io.imread(file_path)
            self.dict.append(image)
            print_progress(i - first + 1, count, prefix="Reading images:", suffix="Complete", bar_length=50)


class RowDescription:
    def __init__(self, row_number, words, digits):
        self.row_number = row_number
        self.words = words
        self.digits = digits


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def angles_from_contours(contour):
    """
    Calculate corner angle from contours.
    :param contour: contours to calculate the angle from
    :return: Angle
    """
    ret = np.zeros(len(contour) - 1)
    for i in range(len(contour) - 1):
        a = 0
        b = 0
        if i == 0:
            a = math.atan2(contour[i + 1, 0] - contour[i, 0], contour[i + 1, 1] - contour[i, 1])
            b = math.atan2(contour[i - 2, 0] - contour[i, 0], contour[i - 2, 1] - contour[i, 1])
        else:
            a = math.atan2(contour[i + 1, 0] - contour[i, 0], contour[i + 1, 1] - contour[i, 1])
            b = math.atan2(contour[i - 1, 0] - contour[i, 0], contour[i - 1, 1] - contour[i, 1])
        if a < 0:
            a = 2 * math.pi - abs(a)
        if b < 0:
            b = 2 * math.pi - abs(b)

        ret[i] = a - b
    return ret


def make_margin(image, margin):
    for i in range(image.shape[1]):
        for ii in range(image.shape[0]):
            if (i < margin) or (ii < margin) or (i > image.shape[1] - margin - 1) or (ii > image.shape[0] - margin - 1):
                image[ii, i] = 0
    return image


def crossing_points(image, angles, distances):
    x = []
    y = []

    for i in range(len(angles)):
        for ii in range(i + 1, len(angles)):
            ctg_t1 = np.tan(np.pi / 2 - angles[i])
            ctg_t2 = np.tan(np.pi / 2 - angles[ii])
            sin_t1 = np.sin(angles[i])
            sin_t2 = np.sin(angles[ii])
            d1 = distances[i]
            d2 = distances[ii]
            x.append((d2 / sin_t2 - d1 / sin_t1) / (ctg_t1 - ctg_t2))
            y.append(ctg_t1 * x[len(x) - 1] + d1 / sin_t1)

    for i in range(len(x) - 1, -1, -1):
        if (x[i] >= image.shape[1]) or (y[i] >= image.shape[0]) or (x[i] < 0) or (y[i] < 0):
            """x.pop(i)
            y.pop(i)"""
    print(image.shape)
    print(x)
    print(y)


def take_biggest_region(image):
    image = measure.label(image, connectivity=1)
    region_props = measure.regionprops(image)

    max_area = 0
    max_area_index = 0
    for i in region_props:
        if i.area > max_area:
            max_area = i.area
            max_area_index = i.label

    image = image == max_area_index

    return image


def take_rectangle_contour(contour, image):
    image_width = image.shape[1]
    image_height = image.shape[0]

    angles = angles_from_contours(contour)
    angle_diff = []
    for i in angles:
        diff = abs((i + np.pi / 2) - np.pi * round((i + np.pi / 2) / np.pi))
        angle_diff.append(diff)
    contour_quarter = []
    for i in contour:
        if (i[0] < image_height / 2) and (i[1] > image_width / 2):
            contour_quarter.append(0)
        elif (i[0] > image_height / 2) and (i[1] > image_width / 2):
            contour_quarter.append(1)
        elif (i[0] > image_height / 2) and (i[1] < image_width / 2):
            contour_quarter.append(2)
        else:
            contour_quarter.append(3)

    new_contour = np.zeros(10).reshape(5, 2)
    best_angle_points = []
    for i in range(4):
        points = []
        for ii in range(len(contour) - 1):
            if contour_quarter[ii] == i:
                points.append(ii)

        best_angle_index = -1
        for ii in points:
            if (best_angle_index == -1) or (angle_diff[best_angle_index] > angle_diff[ii]):
                best_angle_index = ii
        best_angle_points.append(best_angle_index)
        diff = np.pi / 2
        for ii in points:
            if best_angle_index == ii:
                continue
            else:
                if diff > (angle_diff[ii] - angle_diff[best_angle_index]):
                    diff = angle_diff[ii] - angle_diff[best_angle_index]
        if (3 > len(points)) or (diff > np.pi / 4):
            new_contour[i, 0] = int(best_angle_index)
        else:
            new_contour[i, 0] = -1

    aproximated = []
    for i in range(4):
        if new_contour[i, 0] == -1:
            if (new_contour[(i + 1) % 4, 0] != -1) and (new_contour[(i - 1) % 4, 0] != -1):
                point11 = contour[int(new_contour[(i - 1) % 4, 0])]
                point12 = contour[int(new_contour[(i - 1) % 4, 0] + 1)]
                a1 = (point11[1] - point12[1]) / (point11[0] - point12[0])
                b1 = point11[1] - a1 * point11[0]

                point21 = contour[int(new_contour[(i + 1) % 4, 0] - 1)]
                point22 = contour[int(new_contour[(i + 1) % 4, 0])]
                a2 = (point21[1] - point22[1]) / (point21[0] - point22[0])
                b2 = point21[1] - a2 * point21[0]

                new_contour[i, 0] = (b2 - b1) / (a1 - a2)
                new_contour[i, 1] = a1 * new_contour[i, 0] + b1

                if (i == 0) and ((new_contour[i, 0] > image_height / 2) or (new_contour[i, 0] < 0) or (
                        new_contour[i, 1] < image_width / 2) or (new_contour[i, 0] > image_width)):
                    new_contour[i] = contour[best_angle_points[0]]
                elif (i == 1) and ((new_contour[i, 0] > image_height) or (new_contour[i, 0] < image_height / 2) or (
                        new_contour[i, 1] < image_width / 2) or (new_contour[i, 0] > image_width)):
                    new_contour[i] = contour[best_angle_points[1]]
                elif (i == 2) and ((new_contour[i, 0] > image_height) or (new_contour[i, 0] < image_height / 2) or (
                        new_contour[i, 1] < 0) or (new_contour[i, 0] > image_width / 2)):
                    new_contour[i] = contour[best_angle_points[2]]
                elif (i == 3) and ((new_contour[i, 0] > image_height / 2) or (new_contour[i, 0] < 0) or (
                        new_contour[i, 1] < 0) or (new_contour[i, 0] > image_width / 2)):
                    new_contour[i] = contour[best_angle_points[3]]
                aproximated.append(i)

            else:
                new_contour[i, 0] = best_angle_points[i]

    for i in range(4):
        if (new_contour[i, 0] >= 0) and (i not in aproximated):
            new_contour[i] = contour[int(new_contour[i, 0])]

    new_contour[4] = new_contour[0]
    return new_contour


def detect_paper(image):
    # Generating figure 1
    # fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    # ax = axes.ravel()

    grayscale = color.rgb2gray(image)
    # ax[0].imshow(grayscale)

    block_size = 1555
    local_thresh = filters.threshold_local(grayscale, block_size)
    binary_local = grayscale > local_thresh
    # ax[1].imshow(binary_local)

    binary_local = morphology.opening(binary_local, morphology.rectangle(5, 5))
    binary_local = morphology.closing(binary_local, morphology.rectangle(30, 30))

    binary_local = take_biggest_region(binary_local)

    binary_local = make_margin(binary_local, 3)

    contours = measure.find_contours(binary_local, 0.5)
    contour_polygon = measure.approximate_polygon(contours[0], 50)
    paper_contour = take_rectangle_contour(contour_polygon, binary_local)
    # ax[2].imshow(binary_local)
    # for n, contour in enumerate(paper_contour):
    #     ax[2].plot(paper_contour[:, 1], paper_contour[:, 0], linewidth=2)

    # plt.tight_layout()
    # plt.show()

    # Return necessary information for next steps
    return paper_contour


def calculate_distance(first_point, second_point):
    """
    Calculate distance between two points
    :param first_point: first point
    :param second_point: second point
    :return: distance between two points
    """
    return math.sqrt((first_point[0] - second_point[0]) ** 2 + (first_point[1] - second_point[1]) ** 2)


def find_closest_corner(image_shape, input_corner):
    """
    Finds closest corner in the input image for the input corner
    :param image_shape: shape of an input image
    :param countours: location of a calculated corner
    :return: coordinates of the closest corner
    """
    cor_lu = [0, 0]
    cor_ru = [image_shape[0] - 1, 0]
    cor_ld = [0, image_shape[1] - 1]
    cor_rd = [image_shape[0] - 1, image_shape[1] - 1]

    distance = calculate_distance(input_corner, cor_lu)
    chosen_corner = cor_lu

    new_distance = calculate_distance(input_corner, cor_ru)
    if new_distance < distance:
        distance = new_distance
        chosen_corner = cor_ru

    new_distance = calculate_distance(input_corner, cor_ld)
    if new_distance < distance:
        distance = new_distance
        chosen_corner = cor_ld

    new_distance = calculate_distance(input_corner, cor_rd)
    if new_distance < distance:
        chosen_corner = cor_rd

    return chosen_corner


def warp_paper(image, contours):
    """
    Calculates perspective and unwarps paper.
    :param image: base image on which transformation will be used
    :param countours: calculated contours of a sheet of paper
    :param plot: (fig, axes) touple used to manipulate plotting
    """

    (rows, cols, channels) = image.shape

    new_contours = [find_closest_corner((rows, cols), value) for i, value in enumerate(contours)]

    contours = contours[:4]
    new_contours = new_contours[:4]

    contours = np.array(contours)
    new_contours = np.array(new_contours)

    contours[:, [0, 1]] = contours[:, [1, 0]]
    new_contours[:, [0, 1]] = new_contours[:, [1, 0]]

    trans = transform.ProjectiveTransform()
    trans.estimate(new_contours, contours)

    warped_image = transform.warp(image, trans)

    # io.imshow(warped_image)
    # plt.show()

    return warped_image, trans


def remove_paper_noise(image):
    # (fig, axes) = plot
    # ax = axes.ravel()
    """fig, axes = plt.subplots(1, 5, figsize=(14, 6))
    ax = axes.ravel()"""

    grayscale = color.rgb2gray(image)
    grayscale = 1 - grayscale
    grayscale = (grayscale * 255).astype(np.uint8)

    vertical_lines = morphology.opening(grayscale, morphology.rectangle(30, 1))
    horizontal_lines = morphology.opening(grayscale, morphology.rectangle(1, 30))

    lines = vertical_lines * 0.5 + horizontal_lines * 0.5

    grayscale_without_lines = grayscale - lines
    threshold = filters.threshold_yen(grayscale_without_lines)
    binary_without_lines = grayscale_without_lines > threshold

    """fig, ax = try_all_threshold(grayscale_without_lines, figsize=(10, 8), verbose=False)
    plt.show()"""

    """ax[0].imshow(grayscale)
    ax[1].imshow(vertical_lines)
    ax[2].imshow(horizontal_lines)
    ax[3].imshow(grayscale_without_lines)
    ax[4].imshow(binary_without_lines)
    fig.tight_layout()
    plt.show()"""

    return binary_without_lines


def detect_rows(image):
    # fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    # ax = axes.ravel()

    row_sum = np.sum(image, axis=1)
    result = row_sum > np.mean(row_sum)

    result = morphology.opening(result, np.ones(7))

    labeled_result = measure.label(result)

    rows = np.zeros(2 * np.max(labeled_result)).reshape(np.max(labeled_result), 2)

    label = 0
    for i in range(len(labeled_result)):
        if labeled_result[i] != label:
            if label == 0:
                rows[labeled_result[i] - 1, 0] = i
            else:
                rows[label - 1, 1] = i
        label = labeled_result[i]

    for row in rows:
        row[0] -= 15
        row[1] += 15

    # for row in rows:
    # ax[0].plot(np.array([10, 10, 1400, 1400, 10]), np.array([row[0], row[1], row[1], row[0], row[0]]), linewidth=2)

    # ax[0].imshow(image)
    # ax[1].plot(range(len(row_sum)), row_sum)
    # ax[2].plot(range(len(result)), labeled_result)
    #
    # fig.tight_layout()
    # plt.show()

    # io.imshow(image)
    # plt.show()

    return rows


def detect_words(paper, word_rows):
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # ax = axes.ravel()

    result = morphology.closing(paper, morphology.rectangle(5, 5))

    # ax[0].imshow(paper)
    # ax[1].imshow(paper)

    divided_rows = []

    for i, word_row in enumerate(word_rows):
        one_row = result[int(word_row[0]):int(word_row[1]), :]
        labeled_result = measure.label(one_row)

        letter_regions = []

        for region in measure.regionprops(labeled_result):
            if region.area >= 100:
                letter_regions.append(region.bbox)
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr + int(word_row[0])), maxc - minc, maxr - minr, fill=False,
                                          edgecolor='blue', linewidth=1)
                # ax[1].add_patch(rect)

        letter_regions = sorted(letter_regions, key=lambda x: x[1])

        words = []

        min_row = letter_regions[0][0]
        max_row = letter_regions[0][2]
        word_start = letter_regions[0][1]
        word_end = letter_regions[0][3]

        digits = []

        for j in range(len(letter_regions) - 1):
            region_distance = letter_regions[j + 1][1] - letter_regions[j][3]

            # (start column, end column, start row, end row)
            digits.append((letter_regions[j][1], letter_regions[j][3], letter_regions[j][0] + int(word_row[0]),
                           letter_regions[j][2] + int(word_row[0])))

            if min_row > letter_regions[j][0]:
                min_row = letter_regions[j][0]

            if max_row < letter_regions[j][2]:
                max_row = letter_regions[j][2]

            if region_distance > 30:
                # (start col, end col, start row, end row)
                words.append((word_start, letter_regions[j][3] if letter_regions[j][3] > word_end else word_end,
                              min_row + int(word_row[0]), max_row + int(word_row[0])))
                word_start = letter_regions[j + 1][1]
                min_row = letter_regions[j + 1][0]
                max_row = letter_regions[j + 1][2]
                digits = []
            else:
                word_end = letter_regions[j][3] if letter_regions[j][3] > word_end else word_end

            if j == len(letter_regions) - 2:
                words.append(
                    (word_start, letter_regions[j + 1][3], min_row + int(word_row[0]), max_row + int(word_row[0])))
                digits.append(
                    (letter_regions[j + 1][1], letter_regions[j + 1][3], letter_regions[j + 1][0] + int(word_row[0]),
                     letter_regions[j + 1][2] + int(word_row[0])))

        divided_rows.append(RowDescription(i, words, digits))

        for word in words:
            minc, maxc, minr, maxr = word
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False,
                                      edgecolor='red', linewidth=1)
            # ax[1].add_patch(rect)

    # plt.show()

    return divided_rows


def train_model():
    # Define a transform to normalize the data
    transform_torch = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

    # Download and load the training data
    trainData = datasets.MNIST('./trainData/', download=True, train=True, transform=transform_torch)
    testData = datasets.MNIST('./trainData/', download=True, train=False, transform=transform_torch)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testData, batch_size=64, shuffle=True)

    cnn = ModelNN()
    optimX = optim.SGD(cnn.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochsNumber = 15
    looser = nn.NLLLoss()

    for i in range(epochsNumber):
        actual_loss = 0
        for images, labels in trainLoader:

            images = images.view(images.shape[0], -1)
            optimX.zero_grad()
            output = cnn(images)
            loss = looser(output, labels)
            loss.backward()
            optimX.step()
            actual_loss += loss.item()
        else:
            print("Epoch " + str(i) + " Loss " + str(actual_loss / len(trainLoader)))

    torch.save(cnn.state_dict(), "weights.pt")
    print("\n Training finished after " + str(time() - time0) + " seconds")


def tag_detected_words(image, divided_rows, trans, output_path, image_number):
    new_img = np.zeros(image.shape, dtype=np.uint8)

    for single_row in divided_rows:
        for word in single_row.words:
            start_point = (word[2], word[0])
            end_point = (word[3], word[1])

            rr, cc = draw.rectangle(start_point, end=end_point, shape=new_img.shape)
            new_img[rr, cc] = single_row.row_number + 1

    new_img = transform.warp(new_img, trans.inverse)

    io.imsave(Path(output_path + "/" + str(image_number) + "-wyrazy.png"), new_img)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # ax = axes.ravel()
    # ax[0].imshow(image)
    # ax[1].imshow(new_img)
    # plt.show()


if __name__ == "__main__":
    first = 1

    # if len(sys.argv) > 4:
    #     if sys.argv[3] != '':
    #         first = int(sys.argv[3])
    photos = PhotoDictPIRO(sys.argv[1], int(sys.argv[2]))

    Path(sys.argv[3]).mkdir(parents=True, exist_ok=True)

    i = 0

    train = False

    if train:
        train_model()

    cnn = ModelNN()
    cnn.load_state_dict(torch.load("weights.pt"))
    cnn.eval()

    image_sequential_number = 0

    for image in photos.dict:
        contours = detect_paper(image)
        warped_image, trans = warp_paper(image, contours)
        clean_paper = remove_paper_noise(warped_image)
        rows = detect_rows(clean_paper)
        divided_rows = detect_words(clean_paper, rows)

        # Change 0 to correct image number
        tag_detected_words(image, divided_rows, trans, sys.argv[3], image_sequential_number)

        file_rows_list = []
        for row in divided_rows:
            digits = row.digits

            row_digits_list = []
            for digit in digits:
                i += 1
                digit_img = take_biggest_region(clean_paper[digit[2] - 2:digit[3] + 2, digit[0] - 2:digit[1] + 2])
                rows_sum = np.sum(digit_img, axis=1)
                column_sum = np.sum(digit_img, axis=0)
                max_row = -1
                min_row = -1
                max_column = -1
                min_column = -1
                for i in range(digit_img.shape[0]):
                    if (min_row == -1) and (rows_sum[i] != 0):
                        min_row = i
                    if rows_sum[i] != 0:
                        max_row = i
                for i in range(digit_img.shape[1]):
                    if (min_column == -1) and (column_sum[i] != 0):
                        min_column = i
                    if column_sum[i] != 0:
                        max_column = i
                digit_img = digit_img[min_row:max_row, min_column:max_column]

                maxDimension = max(digit_img.shape)
                digit_img_cutted = np.zeros(maxDimension * maxDimension).reshape(maxDimension, maxDimension)
                width = int(digit_img.shape[1])
                height = int(digit_img.shape[0])
                digit_img_cutted[int((maxDimension - height) / 2):int((maxDimension - height) / 2 + height),
                int((maxDimension - width) / 2):int((maxDimension - width) / 2 + width)] = digit_img
                digit_img_resized2 = resize(digit_img_cutted, (20, 20), anti_aliasing=True)
                digit_img_resized2 = morphology.dilation(digit_img_resized2, morphology.rectangle(1, 1))
                digit_img_resized = np.zeros(784).reshape(28, 28)
                digit_img_resized[4:24, 4:24] = digit_img_resized2

                digit_img_resized = digit_img_resized / np.max(digit_img_resized) * 2 - 1
                digit_img_resized = np.expand_dims(digit_img_resized, axis=0)

                digit_view = torch.Tensor(digit_img_resized).view(1, 784)
                with torch.no_grad():
                    logps = cnn(digit_view)
                ps = torch.exp(logps).detach()
                probab = list(ps.numpy()[0])
                # print(probab.index(max(probab)))
                row_digits_list.append(probab.index(max(probab)))

                # fig, axes = plt.subplots(1, 4, figsize=(12, 6))
                # ax = axes.ravel()
                # ax[0].imshow(clean_paper)
                # minc, maxc, minr, maxr = digit
                # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False,
                #                           edgecolor='red', linewidth=1)
                # ax[0].add_patch(rect)
                # ax[1].imshow(digit_img)
                # ax[2].imshow(digit_img_cutted)
                # ax[3].imshow(digit_img_resized.squeeze())
                # """io.imsave("./"+str(i)+".png", digit_img_resized)"""
                # plt.show()

            one_row_string = "".join(str(el) for el in row_digits_list)
            file_rows_list.append(one_row_string)
        one_file_string = "\n".join(file_rows_list)

        # Change 0 to correct image number
        with open(Path(sys.argv[3] + "/" + str(image_sequential_number) + "-indeksy.txt"), "w") as text_file:
            print(one_file_string, file=text_file)

        image_sequential_number += 1
