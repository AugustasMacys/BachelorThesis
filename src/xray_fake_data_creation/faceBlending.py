from glob import glob
import random
from collections import OrderedDict
import argparse, os
from os.path import basename, splitext

from color_transfer import color_transfer
from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import cv2
import dlib
from tqdm import tqdm



def main(margin_changer):
    args = get_parser()
    srcFaces = glob(args.srcFacePath)
    random.shuffle(srcFaces)
    databaseFaces = glob(args.faceDatabase)

    # face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shapePredictor)

    for i, srcFace in enumerate(srcFaces):
        try:
            srcFaceBgr = cv2.imread(srcFace)
            height, width, _ = srcFaceBgr.shape
            margin_width = width // margin_changer
            margin_height = height // margin_changer
            srcFaceBgr = srcFaceBgr[margin_height:height - margin_height, margin_width:width - margin_width]
            srcFaceBgr = cv2.resize(srcFaceBgr, (224, 224))
        except:
            print(f'Fail loading: {srcFace}')
            continue

        # detect landmarks
        srcLms = get_landmarks(detector, predictor, cv2.cvtColor(srcFaceBgr, cv2.COLOR_BGR2RGB))
        if srcLms is None:
            print(f'No face: {srcFace}')
            continue
        # find first face whose landmarks are close enough in real face database
        targetBgr, targetLandmarks = find_one_neighbor(detector, predictor, srcFace, srcLms, databaseFaces,
                                                       args.threshold)
        if targetBgr is None:  # if not found
            print(f'No Match: {srcFace}')
            continue

        hullMask = convex_hull(srcFaceBgr.shape, srcLms)  # size (h, w, c) mask of face convex hull
        # generate random deform
        anchors, deformedAnchors = random_deform(hullMask.shape[:2], 4, 4)
        # piecewise affine transform and blur
        warped = piecewise_affine_transform(hullMask, anchors, deformedAnchors)  # size (h, w) warped mask
        blured = cv2.GaussianBlur(warped, (5, 5), 3)

        targetBgrT = color_transfer(srcFaceBgr, targetBgr)
        resultantFace = forge(srcFaceBgr, targetBgrT, blured)  # forged face

        # save face images
        final_mask = 4.0 * np.multiply(blured, (1.0 - blured))
        final_mask = (final_mask * 255).astype(np.uint8)
        _, final_mask = cv2.threshold(final_mask, 1, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'./masks_new/{margin_changer}blured_{i}.png', final_mask)
        cv2.imwrite(f'./forged_new/{margin_changer}forge_{i}.png', resultantFace)

    return


def get_landmarks(detector, predictor, rgb):
    # first get bounding box (dlib.rectangle class) of face.
    boxes = detector(rgb, 1)
    for box in boxes:
        landmarks = shape_to_np(predictor(rgb, box=box))
        break
    else:
        return None
    return landmarks.astype(np.int32)


def find_one_neighbor(detector, predictor, srcPath, srcLms, faceDatabase, threshold):
    random.shuffle(faceDatabase)
    for i, face in enumerate(faceDatabase):
        if i == 1000:
            threshold = 100
        if i == 2500:
            threshold = 200
        try:
            rgb = cv2.imread(face)
            height, width, _ = rgb.shape
            margin_width = width // 5
            margin_height = height // 5
            rgb = rgb[margin_height:height - margin_height, margin_width:width - margin_width]
            rgb = cv2.resize(rgb, (224, 224))
        except:
            continue
        rgb = dlib.resize_image(rgb, 224, 224)
        landmarks = get_landmarks(detector, predictor, rgb)
        if landmarks is None:
            continue
        dist = distance(srcLms, landmarks)
        if dist < threshold and basename(face).split('_')[0] != basename(srcPath).split('_')[0]:
            return rgb, landmarks
    return None


def forge(srcRgb, targetRgb, mask):
    return (mask * targetRgb + (1 - mask) * srcRgb).astype(np.uint8)


def convex_hull(size, points, fillColor=(255,) * 3):
    mask = np.zeros(size, dtype=np.uint8)  # mask has the same depth as input image
    points = cv2.convexHull(np.array(points))
    corners = np.expand_dims(points, axis=0).astype(np.int32)
    cv2.fillPoly(mask, corners, fillColor)
    return mask


def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    h, w = imageSize
    rows = np.linspace(0, h - 1, nrows).astype(np.int32)
    cols = np.linspace(0, w - 1, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:, 0], 0, h - 1, deformed[:, 0])
    np.clip(deformed[:, 1], 0, w - 1, deformed[:, 1])
    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    warped = warp(image, trans)
    return warped


def distance(lms1, lms2):
    return np.linalg.norm(lms1 - lms2)


FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_parser():
    parser = argparse.ArgumentParser(description='Demo for face x-ray fake sample generation')
    parser.add_argument('--srcFacePath', '-sfp', type=str)
    parser.add_argument('--faceDatabase', '-fd', type=str)
    parser.add_argument('--threshold', '-t', type=float, default=70, help='threshold for facial landmarks distance')
    parser.add_argument('--shapePredictor', '-sp', type=str, default='shape_predictor_68_face_landmarks.dat',
                        help='Path to dlib facial landmark predictor model')
    return parser.parse_args()


if __name__ == '__main__':
    for i in range(4, 9):
        main(i)
