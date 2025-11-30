import os
import numpy as np
import rasterio
import cv2


#Find B02/B03/B04
def find_rgb_bands_from_imgdata(folder):
    bands = {"B02": None, "B03": None, "B04": None}

    for f in os.listdir(folder):
        fl = f.lower()
        full = os.path.join(folder, f)

        if not fl.endswith(".jp2"):
            continue

        if "b02_10m" in fl:
            bands["B02"] = full
        if "b03_10m" in fl:
            bands["B03"] = full
        if "b04_10m" in fl:
            bands["B04"] = full

    return bands


# Read RGB
def read_rgb_from_bands(bands):
    # check if 3 channels are available
    for b in ["B02", "B03", "B04"]:
        if bands[b] is None:
            raise FileNotFoundError(f"Not found {b}")

    # read them with rasterio
    with rasterio.open(bands["B04"]) as r:
        R = r.read(1).astype(np.float32)
    with rasterio.open(bands["B03"]) as g:
        G = g.read(1).astype(np.float32)
    with rasterio.open(bands["B02"]) as b:
        B = b.read(1).astype(np.float32)

    rgb = np.dstack([R, G, B])# all 3 channels in one RGB

    # stretching the histogram to improve image contrast
    p1, p99 = np.percentile(rgb, (1, 99))
    rgb = np.clip((rgb - p1) / (p99 - p1), 0, 1) * 255
    return rgb.astype(np.uint8)


#Image enhancement for processing
def enhance_image(rgb):
    img = rgb.astype(np.float32) / 255.0
    img = img ** 0.7
    img = (img * 255).clip(0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    #Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
    L2 = clahe.apply(L)

    lab2 = cv2.merge((L2, A, B))
    res = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)


#Scaling image sizes
def resize_to_same(im1, im2):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    target_w = min(w1, w2)

    im1r = cv2.resize(im1, (target_w, int(h1 * target_w / w1)))
    im2r = cv2.resize(im2, (target_w, int(h2 * target_w / w2)))

    h = min(im1r.shape[0], im2r.shape[0])

    return im1r[:h], im2r[:h]

#Oriented FAST and Rotated BRIEF, filtering and matching points
def match_and_draw(im1, im2, max_draw=300):
    orb = cv2.ORB_create(nfeatures=6000)

    kp1, d1 = orb.detectAndCompute(im1, None)
    kp2, d2 = orb.detectAndCompute(im2, None)

    # KNN matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(d1, d2, 2)

    #Lowe ratio test
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return None, 0

    #homography
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    mask = mask.ravel().tolist()

    #Filtering inliers only
    inliers = [m for i, m in enumerate(good) if mask[i] == 1]
    inliers = sorted(inliers, key=lambda x: x.distance)[:max_draw]

    #Draw matches
    matched = cv2.drawMatches(
        im1, kp1,
        im2, kp2,
        inliers, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return matched, len(inliers)