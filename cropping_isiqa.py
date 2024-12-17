from assets import rectangling
import os
import cv2 as cv


if __name__ == "__main__":
    dir_names = os.listdir("output_stitched/adobe_panoramas")
    for dir_name in dir_names:
        if dir_name is "office":
            if not dir_name.endswith(".csv"):
                subset_names = os.listdir(f"output_stitched/isiqa/{dir_name}")
                for subset in subset_names:
                    try:
                        if "cropped" not in subset and os.path.exists(f"output_stitched/isiqa/{dir_name}/{subset}"):
                            raw_img = cv.imread(f"output_stitched/isiqa/{dir_name}/{subset}")
                            cropped_f_name = f"output_stitched/isiqa/{dir_name}/{subset[:-4]}_cropped.png"
                            cropped_img = rectangling(raw_img)
                            if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                                cv.imwrite(cropped_f_name, cropped_img)
                                print(f"Cropped: {cropped_f_name}")
                            else:
                                print(f"Failed to Crop: {cropped_f_name}")
                    except cv.error as e:
                        print(e)
