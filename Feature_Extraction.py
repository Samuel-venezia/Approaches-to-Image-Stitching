import os
import tarfile
import cv2 as cv
import numpy as np
import datetime
from assets import find_sub_image, rectangling, crop_img
from homography import homography
import scipy.spatial.distance as distance


# region Current Classes
class Images:
    def __init__(self, image_1=None, image_2=None):
        if type(image_1) is np.ndarray:
            self.__images = [image_1]
        else:
            self.__images = []
        if type(image_2) is np.ndarray:
            self.__images.append(image_2)

    def add_image(self, image):
        if len(self.__images) < 2:
            self.__images.append(image)
        else:
            print("Max images stored. Please remove an image from the list or clear the list")

    def reset_images(self):
        self.__images = []

    def get_images(self):
        point_images = []
        for image in self.__images:
            point_images.append(np.copy(image))
        return point_images

    def get_images_gray(self):
        gray_images = []
        for image in self.__images:
            gray_images.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        return gray_images

    def remove_image(self, image):
        self.__images.remove(image)

    def scaling(self, scale_per=1.0):
        for image in self.__images:
            self.__images[self.__images.index(image)] = cv.resize(image, None, fy=scale_per, fx=scale_per,
                                                                  interpolation=cv.INTER_LINEAR)

    def show_images(self):
        for index, image in enumerate(self.__images):
            cv.imshow(f"Image {index + 1}", image)
        cv.waitKey(0)


class Detectors:
    # This class will deal with the feature detectors that will be implemented
    def __init__(self, images):
        self.__keypoints = []
        self.__images = images

    def get_keypoints(self):
        return self.__keypoints

    def sift(self):
        sift = cv.SIFT_create()
        for index, image in enumerate(self.__images):
            self.__keypoints.append(sift.detect(image, None))
        return self.__keypoints

    def orb(self):
        orb = cv.ORB_create()
        for index, image in enumerate(self.__images):
            self.__keypoints.append(orb.detect(image, None))

    def brisk(self):
        brisk = cv.BRISK_create()
        for image in self.__images:
            self.__keypoints.append(brisk.detect(image, None))

    def akaze(self):
        akaze = cv.AKAZE_create()
        for image in self.__images:
            self.__keypoints.append(akaze.detect(image, None))

    def surf(self):
        surf = cv.xfeatures2d.SURF_create()
        for image in self.__images:
            self.__keypoints.append(surf.detect(image, None))

    def fast(self):
        fast_detector = cv.FastFeatureDetector_create()
        for index, image in enumerate(self.__images):
            # Can work in both colour and greyscale
            self.__keypoints.append(fast_detector.detect(image, None))
            self.__images[index] = cv.drawKeypoints(image, self.__keypoints[index], None, flags=0)
            # cv.imshow(f"Image {index+1} FAST Feature Detection", self.__images[index])


class Description:
    # This class will implement the second stage of the pipeline - Feature Descriptors
    def __init__(self, images, key_points):
        self.__images = images
        self.__keypoints = key_points
        self.__descriptors = []

    def get_descriptors(self):
        return self.__descriptors

    def sift(self):
        sift = cv.SIFT_create()
        _, self.__descriptors = sift.compute(self.__images, self.__keypoints)

    def surf(self):
        surf = cv.xfeatures2d.SURF_create()
        for index, image in enumerate(self.__images):
            _, descriptors = surf.compute(image, self.__keypoints[index])
            self.__descriptors.append(descriptors)

    def brisk(self):
        brisk = cv.BRISK_create()
        _, self.__descriptors = brisk.compute(self.__images, self.__keypoints)

    def orb(self):
        # This isn't a great way to write this. The same justification as ORB, there isn't another way of writing
        # this that results in a successful program
        # The error message for this was that there had been a memory error
        orb = cv.ORB_create()
        for index, image in enumerate(self.__images):
            _, descriptor = orb.detectAndCompute(image, None)
            self.__descriptors.append(descriptor)

    def akaze(self):
        # This isn't a great way to write this.
        # Using Detect and Compute will most likely increase the processing time, however,
        # there isn't another way of doing it.
        # I've tried an alternative approach - the same approach as the other descriptors, but it doesn't work
        akaze = cv.AKAZE_create()
        for index, image in enumerate(self.__images):
            _, self.__descriptors = akaze.detectAndCompute(image, None)

    def brief(self):
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        _, self.__descriptors = brief.compute(self.__images, self.__keypoints)

    def freak(self):
        freak = cv.xfeatures2d.FREAK_create()
        _, self.__descriptors = freak.compute(self.__images, self.__keypoints)


class Matching:
    # This class will implement the third stage of the pipeline, feature matching -  matching pairs
    def __init__(self, images, key_points, descriptors):
        self.__images = images
        self.__matches = None
        self.__keypoints = key_points
        self.__descriptors = descriptors

    def get_matched_pairs(self):
        return self.__matches

    def flann(self):
        flann_matcher = cv.FlannBasedMatcher_create()
        try:
            self.__matches = flann_matcher.knnMatch(np.float32(self.__descriptors[0]),
                                                    np.float32(self.__descriptors[1]), 2)
        except cv.error as err:
            print(err)
            pass
        passed, good = [], []
        for match, nearest in self.__matches:
            if match.distance < 0.75 * nearest.distance:
                passed.append([match])
                good.append(match)
        self.__matches = good

    def brute_force(self, batch_distance=cv.NORM_L1):
        try:
            bfm = cv.BFMatcher(batch_distance, crossCheck=True)
            self.__matches = bfm.match(self.__descriptors[0], self.__descriptors[1])
            self.__matches = sorted(self.__matches, key=lambda x: x.distance)
        except cv.error:
            self.brute_force(batch_distance=cv.NORM_HAMMING)
        # draw image to visualise the matches between the key points utilising the descriptors from stage 2.
        # point_images = self.__images
        # matches_comparison_image = cv.drawMatches(point_images[0], self.__keypoints[0], point_images[1],
        #                                                  self.__keypoints[1], self.__matches, None)

    def brute_force_knn(self):
        bfm = cv.BFMatcher()
        self.__matches = bfm.knnMatch(self.__descriptors[0], self.__descriptors[1], k=2)
        # Ratio Test
        passed, good = [], []
        for match, nearest in self.__matches:
            if match.distance < 0.75 * nearest.distance:
                passed.append([match])
                good.append(match)
        self.__matches = good
        # draw image to visualise the matches between the key points utilising the descriptors from stage 2.
        # point_images = self.__images
        # self.__matches_comparison_image = cv.drawMatchesKnn(point_images[0], self.__keypoints[0], point_images[1],
        #                                                     self.__keypoints[1], passed, None)


class HomographyEstimation:
    # This class will implement the last stage oif the main pipeline - The inlier/outlying matched pairs
    def __init__(self, images, key_points, matches):
        self.__images = images
        self.__keypoints = key_points
        self.__matches = matches
        self.__mtx, self.__mask, self.__matches_draw, self.__orig_pts_mtx, self.__dest_pts_mtx = None, None, \
            None, None, None

    def orig_dest_mtx(self):
        # Need 4 pairs in order to perform homography
        min_max_count = 4
        if len(self.__matches) >= min_max_count:
            self.__orig_pts_mtx = np.float32([self.__keypoints[0][m.queryIdx].pt for m in self.__matches]).reshape(-1,
                                                                                                                   1, 2)
            self.__dest_pts_mtx = np.float32([self.__keypoints[1][m.trainIdx].pt for m in self.__matches]).reshape(-1,
                                                                                                                   1, 2)

    def dl_orig_dest_mtx(self):
        # Need 4 pairs in order to perform homography
        min_max_count = 4
        if len(self.__matches) >= min_max_count:
            self.__orig_pts_mtx = np.float32([m[0] for m in self.__matches]).reshape(-1, 1, 2)
            self.__dest_pts_mtx = np.float32([m[1] for m in self.__matches]).reshape(-1, 1, 2)

    def get_homography_matrix(self):
        return self.__mtx

    def set_homography_matrix(self, mtx):
        self.__mtx = mtx

    def set_inverse_homography_matrix(self, h_mtx):
        self.__mtx = np.linalg.inv(h_mtx)

    def ransac(self):
        self.__mtx, self.__mask = cv.findHomography(self.__orig_pts_mtx, self.__dest_pts_mtx, cv.RANSAC, 5.0)
        pass

    def prosac(self):
        self.__mtx, self.__mask = cv.findHomography(self.__orig_pts_mtx, self.__dest_pts_mtx, cv.RHO, 5.0)

    def usac(self):
        self.__mtx, self.__mask = cv.findHomography(self.__orig_pts_mtx, self.__dest_pts_mtx, cv.USAC_DEFAULT, 5.0)

    def magsac(self):
        self.__mtx, self.__mask = cv.findHomography(self.__orig_pts_mtx, self.__dest_pts_mtx, cv.USAC_MAGSAC, 5.0)

    def perspective_warping(self):
        height, width, _ = self.__images[0].shape
        if self.__mtx is not None:
            # warping the perspective requires a 3x3 transformational matrix (mtx)
            h1, w1 = self.__images[0].shape[:2]
            h2, w2 = self.__images[1].shape[:2]
            pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            pts2_ = cv.perspectiveTransform(pts2, self.__mtx)
            pts = np.concatenate((pts1, pts2_), axis=0)
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel())
            translate = [-xmin, -ymin]
            homography_translate = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
            result = cv.warpPerspective(self.__images[0], homography_translate.dot(self.__mtx),
                                        (xmax - xmin, ymax - ymin))
            image_1_translate = np.copy(result)
            image_1_translate[translate[1]:h2 + translate[1], translate[0]:w2 + translate[0]] = self.__images[1]
            # Stitched, Warped
            return image_1_translate, result
        else:
            # Unable to generate stitched image. Don't have enough matched pairs
            return None, None


class Pipeline(Images):
    # This class has been designed to encapsulate the four main stages of the pipeline.
    def __int__(self, image_1, image_2):
        self.__key_points, self.__descriptors, self.__matches, self.__warped, self.__stitched = None, None, None, None, None
        self.__homography_est = None
        self.__duration = 0
        super().__init__(image_1, image_2)

    def stage_1(self, algorithm_chosen="SIFT"):
        detector = Detectors(self.get_images_gray())
        if algorithm_chosen == "ORB":
            detector.orb()
        elif algorithm_chosen == "BRISK":
            detector.brisk()
        elif algorithm_chosen == "AKAZE":
            detector.akaze()
        elif algorithm_chosen == "FAST":
            detector.fast()
        else:
            detector.sift()
        self.__key_points = detector.get_keypoints()

    def stage_2(self, algorithm_chosen="SIFT"):
        descriptor = Description(self.get_images_gray(), self.__key_points)
        if algorithm_chosen == "ORB":
            descriptor.orb()
        elif algorithm_chosen == "BRISK":
            descriptor.brisk()
        elif algorithm_chosen == "AKAZE":
            descriptor.akaze()
        elif algorithm_chosen == "BRIEF":
            descriptor.brief()
        elif algorithm_chosen == "FREAK":
            descriptor.freak()
        else:
            descriptor.sift()
        self.__descriptors = descriptor.get_descriptors()

    def stage_3(self, algorithm_chosen="BF"):
        matched_pairs = Matching(self.get_images_gray(), self.__key_points, self.__descriptors)
        if algorithm_chosen == "BF KNN":
            matched_pairs.brute_force_knn()
        elif algorithm_chosen == "FLANN":
            matched_pairs.flann()
        else:
            matched_pairs.brute_force()
        self.__matches = matched_pairs.get_matched_pairs()
        if len(self.__matches) < 4:
            raise LookupError("Need more points for homography")

    def warped_stitched_images(self, algorithm_chosen="RANSAC", inverse=False):
        self.__homography_est = HomographyEstimation(self.get_images(), self.__key_points, self.__matches)
        self.__homography_est.orig_dest_mtx()
        if algorithm_chosen == "USAC":
            self.__homography_est.usac()
        elif algorithm_chosen == "MAGSAC":
            self.__homography_est.magsac()
        elif algorithm_chosen == "PROSAC":
            self.__homography_est.prosac()
        else:
            self.__homography_est.ransac()
        timer_start = datetime.datetime.now()
        if inverse:
            self.__homography_est.set_inverse_homography_matrix(self.__homography_est.get_homography_matrix())
        stitched, self.__warped = self.__homography_est.perspective_warping()
        try:
            self.__stitched = stitched
            self.__duration = datetime.datetime.now() - timer_start
        except AttributeError as e:
            print(f"Att Error: {e}")
            self.__stitched = None
            self.__duration = "N/A"

    def get_homography(self):
        return self.__homography_est.get_homography_matrix()

    def get_stitched_image(self):
        # timer_start = datetime.datetime.now()
        cropped = crop_img(self.__stitched)
        # self.__duration += (datetime.datetime.now() - timer_start)
        return self.__stitched

    def get_warped_image(self):
        return self.__warped

    def get_rectangled_image(self):
        # rectangled_start = datetime.datetime.now()
        rectangled = rectangling(self.__stitched)
        # self.__duration += (datetime.datetime.now() - rectangled_start)
        return rectangled

    def get_duration(self):
        return self.__duration


# endregion


def read_graf(file_name="graf.tar.gz", path_name="img_homography", file_type=".ppm"):
    images_raw = []
    with tarfile.open(file_name, "r") as t:
        print(t.getmembers())
        for member in t.getmembers():
            if file_type in member.path or "to" in member.path or "H_" in member.path:
                file_name = f"{path_name}/{member.path}"
                images_raw.append(file_name)
                # if this image in path already exists then, don't extract.
                if not os.path.isfile(file_name):
                    t.extract(member.name, path_name)
    return images_raw


def formatting_images(stitched_experimental, warped_image, roi_img):
    experimental_top_left = find_sub_image(stitched_experimental, roi_img)
    return warped_image[experimental_top_left[1]:experimental_top_left[1] + roi_img.shape[0],
           experimental_top_left[0]:experimental_top_left[0] + roi_img.shape[1], :]


def cropping_img(img):
    for index, row in enumerate(img):
        if np.count_nonzero(row) > 0:
            return img.shape[0] - index


def format_df(df, starting_column):
    formatted_df = df.copy()
    formatted_df = formatted_df.replace('N/A', np.NAN)
    column_names = []
    for column_number in range(len(formatted_df.columns)):
        column_names.append(f"Image {column_number + starting_column}")
    formatted_df.columns = column_names
    formatted_df["Mean"] = formatted_df.values.mean(axis=1)
    return formatted_df
