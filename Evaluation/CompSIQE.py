import sys
import cv2 as cv
import numpy as np
from scipy.stats import entropy
import math
import assets

# C = 2.5
# a1 = 128
# a2 = 64
# b1 = 10
# b2 = 100
# According to Wojciech Chlewicki, the constants for their 2021 and subsequent 2022 paper are:
C = 0.0000001
a1 = 1.0
a2 = 0.5
b1 = 0.1
b2 = 0.6


class MIQM:
    def __init__(self, imgs):
        self.__s_img = None
        # The Stitched image needs to be cropped for each of the constituent images that are being used as an evaluation
        # This is done by finding the image in which was distorted within the stitched image.
        # to do this, I'll utilise the find_sub_image function from the assets file.
        # As the MIQM is a full-reference IQA metric, smaller, specific parts of the constituent image needs to be used.
        # This changes the evaluation metric to only looking at and evaluating sub-sections of the stitched image,
        # rather than the image as a whole. This is explained on page 9 of Okarma and Kopyek 2022,
        # "Improved Metric for Automatic Quality Assessment of Stitched Images"
        self.__r_img, self.__c_macroblock, self.__s_macroblock, self.__s_imgs = None, [], [], []
        self.__c_imgs = imgs

    def set_s_img(self, s_img):
        self.__s_img = s_img

    def reference_imgs(self):
        for img in self.__c_imgs:
            roi_width, roi_height = int(img.shape[1] * 0.05), int(img.shape[0] * 0.05)
            # the roi is approx 10% the size of the actual image.
            roi_start_x, roi_start_y = int(img.shape[1] / 2) - roi_width, int(img.shape[0] / 2) - roi_height
            # Getting the custom ROI for each of the constituent images
            top_left_s_img = assets.find_sub_image(self.__s_img, img[roi_start_y:roi_start_y + roi_height,
                                                                 roi_start_x:roi_start_x + roi_width, :])
            self.__s_imgs.append(self.__s_img[top_left_s_img[1]:top_left_s_img[1] + roi_height,
                                 top_left_s_img[0]:top_left_s_img[0] + roi_width, :])
        for s_img in self.__s_imgs:
            self.__s_macroblock.append(self.__gen_macroblock(cv.cvtColor(s_img, cv.COLOR_BGR2GRAY)))

    def set_con_imgs(self):
        for img in self.__c_imgs:
            self.__c_macroblock.append(self.__gen_macroblock(cv.cvtColor(img, cv.COLOR_BGR2GRAY)))

    def edge_structural_index(self, con_img_index=0):
        tri_j = self.__texture_randomness_index(self.__s_macroblock[con_img_index])
        tri_i = self.__texture_randomness_index(self.__c_macroblock[con_img_index])
        edg_si, min_xs = 0, sys.maxsize
        min_length = min(len(tri_i), len(tri_j))
        for m in range(min_length):
            min_x = min(len(tri_i[m]), len(tri_j))
            for n in range(min_x):
                edg_si = 1 - (abs(tri_i[m][n] - tri_j[m][n] / tri_j[m][n]))
            min_xs = min(min_xs, min_x)
        # finding the smallest image, preventing an index error
        edg_si = edg_si / (min_length * min_xs)
        # edg_si_var = np.var(edg_si)
        return edg_si

    def luminance_comparison(self, con_img_index=0):
        mu_j = self.__mu_img_macro(self.__s_imgs[con_img_index])
        mu_i = self.__mu_img_macro(self.__c_imgs[con_img_index])
        std_j = self.__sd_img_macro(self.__s_imgs[con_img_index])
        std_i = self.__sd_img_macro(self.__c_imgs[con_img_index])
        luminance = []
        contrast = []
        for m in range(min(len(mu_i), len(mu_j))):
            l_row, c_row = [], []
            for n in range(min(len(mu_i[m]), len(mu_j[m]))):
                l_row.append(
                    (2 * mu_i[m][n] * mu_j[m][n] + C) / (math.pow(mu_i[m][n], 2) + math.pow(mu_j[m][n], 2) + C))
                c_row.append(
                    (2 * std_i[m][n] * std_j[m][n] + C) / (math.pow(std_i[m][n], 2) + math.pow(std_j[m][n], 2) + C))
            luminance.append(l_row)
            contrast.append(c_row)

        sig_k_t = []
        sig_t = []
        tri = self.__texture_randomness_index(self.__c_macroblock[con_img_index])
        for m in range(min(len(std_i), len(std_j))):
            for n in range(min(len(std_i[m]), len(std_j[m]))):
                sig_k_t.append(luminance[m][n] * contrast[m][n] * tri[m][n])
                sig_t.append(tri[m][n])

        lc = sum(sig_k_t) / sum(sig_t)
        return lc

    # region supporting, private functions
    def __gen_macroblock(self, img):
        macroblocks = []
        for m in range(0, img.shape[0], 21):
            row = []
            for n in range(0, img.shape[1], 21):
                row.append(img[m: m + 21, n: n + 21])
            macroblocks.append(row)
        return macroblocks

    def __bin_macroblock(self, macroblock):
        _, edge_binary = cv.threshold(cv.Canny(macroblock, 100, 200), 50, 255, cv.THRESH_BINARY)
        return edge_binary

    def __texture_randomness_index(self, img_mb):
        text_rand_index, text_rand = [], []
        for m in range(len(img_mb)):
            rand_index_n = []
            for n in range(len(img_mb[m])):
                mu_m_n = np.mean(img_mb[m][n])
                mu_m_n_b = np.mean(self.__bin_macroblock(img_mb[m][n]))
                rand_index_n.append(mu_m_n * mu_m_n_b)
            for ri in rand_index_n:
                if b1 <= ri < b2:
                    tri = a1 + (0.5 * a1 * ((math.log2(ri)) / math.log2(b1)))
                elif ri >= b2:
                    tri = a2 + (0.5 * a2 * math.pow(2, (-ri - b2)))
                else:
                    tri = a1
                text_rand_index.append(tri)
            text_rand.append(text_rand_index)
        return text_rand

    def __mu_img_macro(self, img):
        mu_row = []
        macro_b = self.__gen_macroblock(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        for m in range(len(macro_b)):
            mu = []
            for mb in macro_b[m]:
                mu.append(np.mean(mb))
            mu_row.append(mu)
        return mu_row

    def __sd_img_macro(self, img):
        sd_row = []
        macro_b = self.__gen_macroblock(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        for m in range(len(macro_b)):
            sd = []
            for mb in macro_b[m]:
                sd.append(np.std(mb))
            sd_row.append(sd)
        return sd_row
    # endregion


class Entropy:
    def __init__(self, img):
        if len(img.shape) == 3:
            self.__img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            self.__img_grey = img

    def global_ent(self):
        # number of categories/bins is 256
        return self.__ent(self.__img_grey)

    def local_ent(self):
        local_ent = []
        for y in range(0, self.__img_grey.shape[0], 1):
            for x in range(0, self.__img_grey.shape[1] - 9, 1):
                if y + 9 >= self.__img_grey.shape[0]:
                    break
                local_ent.append(self.__ent(self.__img_grey[y:y + 9, x:x + 9]))
        # mean, variance
        return np.mean(local_ent), np.var(local_ent)

    def __ent(self, img):
        # number of bins is 256 according to "Entropy-Based Combined Metric for Automatic Objective
        # Quality Assessment of Stitched Panoramic Images"
        bins = 256
        hist, _ = np.histogram(img.ravel(), bins=bins, range=(0, bins))
        # The entropy function produces the full value of equation 6 from
        # "Entropy-Based Combined Metric for Automatic Objective Quality Assessment of Stitched Panoramic Images"
        # There is no need to reimplement it.
        return entropy((hist / hist.sum()), base=2)


# class to perform the features 2 - 8 of the CompSIQE from
# "Improved Combined Metric for Automatic Quality Assessment of Stitched Images"
class CombSIQE:
    def __init__(self):
        self.__s_local_ent_avg, self.__s_local_ent_var, self.__stitched_image, self.__miqm = None, None, None, None
        self.__constituent_images = None
        self.__siqe = None
        self.__edge_based_s_i_total, self.__lum_contrast_i_total, self.__c_img_global_entropy, \
        self.__c_local_ent_var_total = [], [], [], []

    def set_siqe(self, siqe):
        self.__siqe = siqe

    def set_s_img(self, s_img):
        self.__stitched_image = s_img
        self.__s_local_ent_avg, self.__s_local_ent_var = Entropy(self.__stitched_image).local_ent()
        self.__miqm.set_s_img(self.__stitched_image)
        self.__miqm.reference_imgs()
        for c_img, constituent_image in enumerate(self.__constituent_images):
            self.__edge_based_s_i_total.append(self.__miqm.edge_structural_index(c_img))
            self.__lum_contrast_i_total.append(self.__miqm.luminance_comparison(c_img))

    def set_c_imgs(self, c_imgs):
        self.__constituent_images = c_imgs
        self.__miqm = MIQM(self.__constituent_images)
        self.__miqm.set_con_imgs()
        for c_img, constituent_image in enumerate(self.__constituent_images):
            self.__c_img_global_entropy.append(Entropy(constituent_image).global_ent())
            _, c_local_ent_var = Entropy(constituent_image).local_ent()
            self.__c_local_ent_var_total.append(c_local_ent_var)

    def feature_2(self):
        s_img_global_entropy = Entropy(self.__stitched_image).global_ent()
        return s_img_global_entropy - np.mean(self.__c_img_global_entropy)

    def feature_3(self):
        return self.__s_local_ent_avg

    def feature_4(self):
        return np.var(self.__c_local_ent_var_total) - self.__s_local_ent_var

    def feature_5(self):
        return np.median(self.__edge_based_s_i_total)

    def feature_6(self):
        return np.median(self.__lum_contrast_i_total)

    def feature_7(self):
        return np.var(self.__edge_based_s_i_total)

    def feature_8(self):
        return np.var(self.__lum_contrast_i_total)

    def __call__(self, *args, **kwargs):
        # This will be used to perform the general formula (equation 4)
        # w, a_i, b_j are subject to optimisation, according to page 8.
        w, a_i, b_j = [], [], []
        for index in range(8):
            w.append(1)
            a_i.append(1)
            b_j.append(1)
        f_2 = self.feature_2()
        f_3 = self.feature_3()
        f_4 = self.feature_4()
        f_5 = self.feature_5()
        f_6 = self.feature_6()
        f_7 = self.feature_7()
        f_8 = self.feature_8()
        print("finished MIQM")
        sigma = ((w[1] * f_2) ** a_i[1] + (w[2] * f_3) ** a_i[2] + (w[3] * f_4) ** a_i[3] + (w[4] * f_5) ** a_i[4] + (
                w[5] * f_6) ** a_i[5] + (w[6] * f_7) ** a_i[6] + (w[7] * f_8) ** a_i[7])
        pi_prod = (f_2 ** b_j[1] * f_3 ** b_j[2] * f_4 ** b_j[3] * f_5 ** b_j[4] * f_6 ** b_j[5] * f_7 ** b_j[
            6] * f_8 **
                   b_j[7])
        if self.__siqe is None:
            return sigma, pi_prod
        # CombSQIE = (weight(1) * SIQE^a1) + sigma + (SIQE^b1 * pi_prod)
        return (w[0] * self.__siqe ** a_i[0]) + sigma + (self.__siqe ** b_j[0] * pi_prod)
