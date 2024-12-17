import cv2 as cv


# raw_image = cv.imread("stitching_img/shanghai003.jpg")
raw_image = cv.imread("synthetic_ds/hpatches-sequences-release/v_birdwoman/2.ppm")
cv.imshow("Abstract", cv.imread("synthetic_ds/hpatches-sequences-release/v_bark/1.ppm"))
cv.imshow("Castle 2", cv.imread("synthetic_ds/hpatches-sequences-release/i_castle/2.ppm"))

cv.waitKey(0)
splitting_image = 2
overlapping_region = 0.10
prior_section = 0
split_images = []
width_section = round(raw_image.shape[1]/splitting_image)
# splitting the width of an image into three sections
for section in range(0, splitting_image):
    left_edge = int(section*width_section)
    right_edge = int((section + 1) * width_section)
    if section > 0:
        left_edge = round(left_edge*(1-overlapping_region))
    if section < splitting_image-1:
        right_edge = round(right_edge*(1+overlapping_region))
    cv.imwrite(f"synthetic_ds/testing_ds/{section}.png", raw_image[:, left_edge:right_edge, :])
