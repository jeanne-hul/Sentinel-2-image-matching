import cv2
from model_creation import (
    find_rgb_bands_from_imgdata,
    read_rgb_from_bands,
    enhance_image,
    resize_to_same,
    match_and_draw
)


def process_and_match(folder1, folder2, out_file="output_match.jpg"):
    bands1 = find_rgb_bands_from_imgdata(folder1)
    bands2 = find_rgb_bands_from_imgdata(folder2)

    rgb1 = read_rgb_from_bands(bands1)
    rgb2 = read_rgb_from_bands(bands2)

    img1 = enhance_image(rgb1)
    img2 = enhance_image(rgb2)

    img1, img2 = resize_to_same(img1, img2)

    result, count = match_and_draw(img1, img2)

    print(f" Number of matches (inliers): {count}")

    cv2.imwrite(out_file, result)
    print(f" Saved: {out_file}")

    return out_file


if __name__ == "__main__":
    IMG1 = r"C:\Users\ZhannaHulia\PycharmProjects\image_matching_sentinel2\S2B_MSIL2A_20250606T082559_N0511_R021_T37UCQ_20250606T112504.SAFE\GRANULE\L2A_T37UCQ_A043090_20250606T083407\IMG_DATA\R10m"
    IMG2 = r"C:\Users\ZhannaHulia\PycharmProjects\image_matching_sentinel2\S2B_MSIL2A_20250927T083629_N0511_R064_T37UCQ_20250927T124722.SAFE\GRANULE\L2A_T37UCQ_A044706_20250927T084254\IMG_DATA\R10m"

    process_and_match(IMG1, IMG2)
