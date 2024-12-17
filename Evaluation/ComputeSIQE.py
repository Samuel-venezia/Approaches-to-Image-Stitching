import os
from datetime import datetime
import cv2 as cv
import pandas as pd
from CompSIQE import CombSIQE
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import fmin
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import time


# This file is about running the independent features of the combined SQIE.
# This will allow me to get the 7 independent features for each of the 264 images of the ISIQA dataset.
def isiqa_combined_features(start_index=1, end_index=26):
    # List denotes the number of generated stitched images for each scene within the ISIQA dataset.
    isiqa_breakdown = [9, 12, 12, 11, 11, 11, 12, 8, 10, 7, 9, 10, 9, 11, 10, 6, 9, 10, 12, 12, 13, 8, 9, 12, 12, 9]
    print("Running ISIQA Evaluation ...")
    for img_subset in range(start_index, end_index + 1):
        imgs_c, sub_set = [], []
        names = os.listdir(f"../isiqa_release/constituent_images/{img_subset}")
        start_timer = datetime.now()
        for c_img in names:
            imgs_c.append(cv.imread(f"../isiqa_release/constituent_images/{img_subset}/{c_img}"))
        eval = CombSIQE()
        # Load constituent images
        eval.set_c_imgs(imgs_c)
        if img_subset > 1:
            img_s = sum(isiqa_breakdown[:img_subset - 1]) + 1
        else:
            img_s = 1
        # Load stitched images
        for img_index in range(isiqa_breakdown[img_subset - 1]):
            eval.set_s_img(cv.imread(f"../isiqa_release/stitched_images/{img_s}.jpg"))
            print(f"CombSIQE for img {img_s}\nSubset: {img_subset}")
            sub_set.append([eval.feature_2(), eval.feature_3(), eval.feature_4(), eval.feature_5(), eval.feature_6(),
                            eval.feature_7(), eval.feature_8()])
            img_s += 1
        print(f"Duration for subset {img_subset}: {datetime.now() - start_timer}")
        subset_features_df = pd.DataFrame(sub_set, columns=["f2", "f3", "f4", "f5", "f6", "f7", "f8"])
        subset_features_df.to_csv(f"../Comb_SIQE/f2-8_{img_subset}.csv")


def adobe_stitched_combined_features(dataset="adobe_panoramas"):
    approaches = ["pipeline-AKAZE-FREAK-BF-RANSAC", "pipeline-AKAZE-BRISK-BF-RANSAC",
                  "pipeline-AKAZE-SIFT-BF KNN-RANSAC", "pipeline-AKAZE-SIFT-BF KNN-USAC", "pipeline-AKAZE-SIFT-BF-USAC",
                  "pipeline-SIFT-SIFT-BF KNN-USAC", "auto", "h_sp_sg"]
    dataset_filenames = os.listdir(f"../{dataset}")
    width = 1024
    print(f"Running {dataset} Stitched Evaluation ...")
    for img_subset_index, img_subset in enumerate(dataset_filenames):
        print(f"Subset: {img_subset}")
        imgs_c, sub_set = [], []
        names = os.listdir(f"../{dataset}/{img_subset}")
        file_names = []
        start_timer = datetime.now()
        for name in names:
            if name.endswith(".png") or name.endswith(".jpg"):
                file_names.append(name)
        for c_img in file_names[:4]:
            img_g = cv.imread(f"../{dataset}/{img_subset}/{c_img}")
            imgs_c.append(img_g)
        evaluation = CombSIQE()
        # Load constituent images
        print("Load constituent images")
        evaluation.set_c_imgs(imgs_c)
        print("Load stitched images")
        # Load stitched images
        for approach in approaches:
            for iscropped in ["", "_rectangled"]:
                img_name = f"../output_stitched/{dataset}/{img_subset}/{approach}-{width}{iscropped}.png"
                if os.path.isfile(img_name):
                    try:
                        evaluation.set_s_img(cv.imread(img_name))
                        print(f"CombSIQE: stitched img {approach}{iscropped}")
                        sub_set.append([evaluation.feature_2(), evaluation.feature_3(), evaluation.feature_4(), evaluation.feature_5(),
                                        evaluation.feature_6(), evaluation.feature_7(), evaluation.feature_8()])
                    except cv.error as e:
                        print(e)
                        print(f"{approach}-{width}{iscropped} did not evaluate")
                        sub_set.append(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                else:
                    print(f"{approach}-{width}{iscropped} does not exist")
                    sub_set.append(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
        print(f"Duration for subset {img_subset}: {datetime.now() - start_timer}")
        subset_features_df = pd.DataFrame(sub_set, columns=["f2", "f3", "f4", "f5", "f6", "f7", "f8"])
        subset_features_df.to_csv(f"../Comb_SIQE/f2-8_{dataset}_{img_subset}.csv")


def pow_siqe(w, f, ex):
    if w * f < 0:
        return math.pow(abs(w * f), ex) * -1
    else:
        return math.pow(abs(w * f), ex)


def combining_siqe(w, a_i, b_j, siqe, f_2, f_3, f_4, f_5, f_6, f_7, f_8):
    # This will be used to perform the general formula (equation 4)
    try:
        sigma = (pow_siqe(w[1], f_2, a_i[1]) + pow_siqe(w[2], f_3, a_i[2]) + pow_siqe(w[3], f_4, a_i[3]) +
                 pow_siqe(w[4], f_5, a_i[4]) + pow_siqe(w[5], f_6, a_i[5]) + pow_siqe(w[6], f_7, a_i[6]) + pow_siqe(w[7], f_8, a_i[7]))
        pi_prod = (pow_siqe(1, f_2, b_j[1]) * pow_siqe(1, f_3, b_j[2]) * pow_siqe(1, f_4, b_j[3]) * pow_siqe(1, f_5, b_j[4]) * pow_siqe(1, f_6, b_j[5]) * pow_siqe(1, f_7, b_j[6]) * pow_siqe(1, f_8, b_j[7]))

        return round(float((w[0] * pow_siqe(1, siqe, a_i[0])) + sigma + (pow_siqe(1, siqe, b_j[0]) * pi_prod)), 4)
    except Exception as e:
        return "NA"


def combsiqes(opt):
    combsiqe = []
    isiqa_breakdown = [9, 12, 12, 11, 11, 11, 12, 8, 10, 7, 9, 10, 9, 11, 10, 6, 9, 10, 12, 12, 13, 8, 9, 12, 12, 9]
    # would be 27 scenes of the ISIQA
    for subset_index in range(1, 27):
        f2_8_df = pd.read_csv(f"../Comb_SIQE/f2-8_{subset_index}.csv")
        f1_df = pd.read_csv(f"../Comb_SIQE/quality_scores_{subset_index}.csv")
        f1s = [float(f) for f in f1_df.columns.tolist()]
        f2s = [float(f) for f in f2_8_df["f2"].tolist()]
        f3s = [float(f) for f in f2_8_df["f3"].tolist()]
        f4s = [float(f) for f in f2_8_df["f4"].tolist()]
        f5s = [float(f) for f in f2_8_df["f5"].tolist()]
        f6s = [float(f) for f in f2_8_df["f6"].tolist()]
        f7s = [float(f) for f in f2_8_df["f7"].tolist()]
        f8s = [float(f) for f in f2_8_df["f8"].tolist()]
        w = [float(value) for value in opt[0].tolist()]
        a_i = [float(value) for value in opt[1].tolist()]
        b_j = [float(value) for value in opt[2].tolist()]
        for f_index in range(isiqa_breakdown[subset_index - 1]):
            combsiqe.append(combining_siqe(w, a_i, b_j, f1s[f_index], f2s[f_index], f3s[f_index], f4s[f_index],
                                           f5s[f_index], f6s[f_index], f7s[f_index], f8s[f_index]))
    return combsiqe


def combsiqes_eval(opt, approaches, dataset="adobe_panoramas"):
    combsiqe_subsets = []
    dataset_filenames = os.listdir(f"../{dataset}")
    for subset_index, subset in enumerate(dataset_filenames):
        combsiqe = []
        f2_8_df = pd.read_csv(f"../Comb_SIQE/f2-8_{subset}.csv")
        f1_df = pd.read_csv(f"../Comb_SIQE/quality_scores_{subset}.csv")
        f1s = [f for f in f1_df.columns.tolist()]
        f2s = [f for f in f2_8_df["f2"].tolist()]
        f3s = [f for f in f2_8_df["f3"].tolist()]
        f4s = [f for f in f2_8_df["f4"].tolist()]
        f5s = [f for f in f2_8_df["f5"].tolist()]
        f6s = [f for f in f2_8_df["f6"].tolist()]
        f7s = [f for f in f2_8_df["f7"].tolist()]
        f8s = [f for f in f2_8_df["f8"].tolist()]
        w = [float(value) for value in opt[0].tolist()]
        a_i = [float(value) for value in opt[1].tolist()]
        b_j = [float(value) for value in opt[2].tolist()]
        for f_index in range(len(approaches)):
            if "N/A" in f1s[f_index] or f2s[f_index] == "N/A" or f3s[f_index] == "N/A" or f4s[f_index] == "N/A" or f5s[f_index] == "N/A" or f6s[f_index] == "N/A" or f7s[f_index] == "N/A" or f8s[f_index] == "N/A":
                combsiqe.append("N/A")
            else:
                # combsiqe.append(float(f1s[f_index]))
                combsiqe.append(combining_siqe(w, a_i, b_j, float(f1s[f_index]), float(f2s[f_index]), float(f3s[f_index]), float(f4s[f_index]),
                                        float(f5s[f_index]), float(f6s[f_index]), float(f7s[f_index]), float(f8s[f_index])))
        combsiqe_subsets.append(combsiqe)
    return combsiqe_subsets


def corr_coeff(predicted, actual):
    srocc = round(spearmanr(a=predicted, b=actual).correlation, 4)
    plcc = round(pearsonr(x=predicted, y=actual).statistic, 4)
    krocc = round(kendalltau(x=predicted, y=actual).correlation, 4)
    print(f"PLCC: {plcc} SROCC: {srocc} KROCC: {krocc}")
    return plcc, srocc, krocc


def combsiqe_coefficients_bf():
    mos_file = open("../isiqa_release/MOS.txt", "r")
    mos = [float(line) for line in mos_file.readlines()]
    output_file_name = f"../Comb_SIQE/combsiqe_weights.csv"
    while not os.path.isfile(output_file_name):
        print("New")
        try:
            # w, a_i, b_j are subject to optimisation, according to page 8.
            w, a_i, b_j = [], [], []
            for index in range(8):
                w.append(round(float(random.randrange(1, 99999999)/10000000), 8))
                a_i.append(round(float(random.randrange(1, 99999999)/10000000), 8))
                b_j.append(round(float(random.randrange(1, 99999999)/10000000), 8))
            weights = fmin(corr_return, x0=np.array([w, a_i, b_j]))
            r_combsiqe = combsiqes(weights.reshape(3, 8))
            print(weights)
            # This if statement is just meant for when using a subset,
            # this will be removed when all subsets have been computed
            plcc, srocc, krocc = corr_coeff(r_combsiqe, mos)
            # This dataframe will have three rows, W, A, and B
            # greater than or equals to the target values.
            # I could just search for values that are equal to the target values of Table 2 from,
            # Improved Combined Metric for Automatic Quality Assessment.
            # This would allow me to find the exact values of weights of the study. - Unfortunately, I was unable to find the same exponents to reproduce the correlation coefficients from the CombSIQE paper
            if srocc == 0.8681 and plcc == 0.8676 and krocc == 0.6837:
                print(f"Correlation found\n Weights: {w} Exponents A: {a_i} Exponents B: {b_j}")
                optamisation_values_df = pd.DataFrame([w, a_i, b_j], columns=["1", "2", "3", "4", "5", "6", "7", "8"])
                # This dataframe will have three rows, W, A, and B
                print(optamisation_values_df.to_string())
                optamisation_values_df.to_csv(f"{output_file_name}")
                break
            if srocc >= 0.82 and plcc >= 0.82 and krocc >= 0.6:
                print(f"Improved Correlation found")
                optamisation_values_df = pd.DataFrame(weights)
                # This dataframe will have three rows, W, A, and B
                # print(optamisation_values_df.to_string())
                optamisation_values_df.to_csv(f"../Comb_SIQE/combsiqe_weights_improved.csv")
        except Exception as e:
            print(e)
            pass


def corr_return(indexes):
    mos_file = open("../isiqa_release/MOS.txt", "r")
    mos = [float(line) for line in mos_file.readlines()]
    r_combsiqe = combsiqes(np.array(indexes))
    # This if statement is just meant for when using a subset,
    # this will be removed when all subsets have been computed
    # srocc = fmin(spearmanr, x0=np.array(combsiqes(np.array([w, a_i, b_j]))), args=(np.array(mos)))
    plcc, srocc, krocc = corr_coeff(r_combsiqe, mos)
    if srocc >= 0.8681 and plcc >= 0.8676 and krocc >= 0.6837:
        optamisation_values_df = pd.DataFrame(indexes.reshape(3,8))
        # This dataframe will have three rows, W, A, and B
        print(optamisation_values_df.to_string())
        optamisation_values_df.to_csv(f"../Comb_SIQE/combsiqe_weights.csv")
    return (1 - plcc) + (1 - srocc) + (1 - krocc)


def read_correlation_coefficients(combsiqe_weights_name="combsiqe_weights"):
    mos_file = open("../isiqa_release/MOS.txt", "r")
    mos = [float(line) for line in mos_file.readlines()]
    if not combsiqe_weights_name.endswith(".csv"):
        file_name = combsiqe_weights_name + ".csv"
    else:
        file_name = combsiqe_weights_name
    opt_df = pd.read_csv(f"../Comb_SIQE/{file_name}").drop(columns=["Unnamed: 0"]).values
    r_combsiqe = combsiqes(opt_df)
    plt.scatter(mos, r_combsiqe, marker="x", color="black")
    plt.xlabel("MOS")
    plt.ylabel("CombSIQE")
    plt.xlim(left=0, right=100)
    plt.ylim(bottom=0)
    plcc, srocc, krocc = corr_coeff(r_combsiqe, mos)
    plt.savefig("SP_CombSIQE_ISIQA.png")
    plt.show()


def adobe_combsiqe_results(combsiqe_weights_name="combsiqe_weights.csv", dataset_name=f"../adobe_panoramas", dataname="Adobe"):
    dataset_filenames = os.listdir(dataset_name)
    if not combsiqe_weights_name.endswith(".csv"):
        file_name = combsiqe_weights_name + ".csv"
    else:
        file_name = combsiqe_weights_name
    opt_df = pd.read_csv(f"../Comb_SIQE/{file_name}").drop(columns=["Unnamed: 0"]).values
    approaches = ["AKAZE-FREAK-BF-RANSAC", "AKAZE-FREAK-BF-RANSAC-cropped",
                  "AKAZE-BRISK-BF-RANSAC", "AKAZE-BRISK-BF-RANSAC-cropped",
                  "AKAZE-SIFT-BF KNN-RANSAC", "AKAZE-SIFT-BF KNN-RANSAC-cropped",
                  "AKAZE-SIFT-BF KNN-USAC", "AKAZE-SIFT-BF KNN-USAC-cropped",
                  "AKAZE-SIFT-BF-USAC", "AKAZE-SIFT-BF-USAC-cropped",
                  "SIFT-SIFT-BF KNN-USAC", "SIFT-SIFT-BF KNN-USAC-cropped",
                  "auto", "auto-cropped", "h_sp_sg", "h_sp_sg-cropped"]
    r_combsiqe = combsiqes_eval(opt_df, approaches)
    adobe_combsiqe_df = pd.DataFrame(r_combsiqe, columns=approaches)
    adobe_combsiqe_df = adobe_combsiqe_df.replace("N/A", np.nan)
    adobe_combsiqe_df["Subset"] = dataset_filenames
    print(adobe_combsiqe_df.to_string())
    adobe_combsiqe_df.to_csv(f"../Comb_SIQE/{dataname}_COMBSIQE_stitched_cropped_orig.csv")
    siqe_mod_mean(adobe_combsiqe_df, approaches)


def siqe_mod_mean(siqe_df, approaches):
    siqe_new_df = pd.DataFrame()
    subsets = siqe_df["Subset"].values.tolist()
    subsets.append("Mean")
    for approach_index in range(0, len(approaches), 2):
        siqe_new_df[approaches[approach_index]] = siqe_df[[approaches[approach_index], approaches[approach_index+1]]].mean(axis=1, skipna=True)
    siqe_new_df = siqe_new_df.head(len(siqe_df["Subset"].values))
    siqe_new_df = pd.concat([siqe_new_df, pd.DataFrame(siqe_new_df.mean()).transpose()], ignore_index=True)
    siqe_new_df = siqe_new_df.replace(np.nan, "N/A")
    siqe_new_df.index = subsets
    siqe_new_df = siqe_new_df.round(4)
    siqe_new_df.to_csv("../Comb_SIQE/Adobe_COMBSIQE_avg_stitched_cropped.csv")
    print(siqe_new_df.to_string())


if __name__ == "__main__":
    # adobe_stitched_combined_features generates features 2-8 of CombSIQE for the stitched images generated by each approach.
    # Feature 1, SIQE, is generated with the MATLAB code from the following repo: https://github.com/pavancm/Stitched-Image-Quality-Evaluator
    adobe_stitched_combined_features()
    adobe_combsiqe_results()

    # Iterates through optimisation process to find the optimum weights and exponents for the CombSIQE -- attempting to replicate the weights and exponents given in CombSIQE's paper.
    # combsiqe_coefficients_bf()
    # Use this function to generate a scatter plot of the CombSIQE against the MOS from ISIQA dataset.
    # read_correlation_coefficients("../Comb_SIQE/combsiqe_weights.csv")
    # Provides evidence to use the CombSIQE over other IQA methods including SIQE alone.
    # The Correlation coefficient results of differing IQA metrics against the MOS of the ISIQA
    # PLCC: 0.7459 SROCC: 0.6976 KROCC: 0.5248 SIQE scores
    # PLCC: 0.7593 SROCC: 0.7196 KROCC: 0.5401 COMBSIQE default 1 scores
    # PLCC: 0.8208 SROCC: 0.8204 KROCC: 0.6246 COMBSIQE Improved scores - Features 1-8

    # PLCC: 0.5015 SROCC: 0.5189 KROCC: 0.3654 COMBSIQE Improved Weighted - with the 6 exponents given.
    # PLCC: 0.3865 SROCC: 0.3363 KROCC: 0.2436 EntSIQE+1 with exponents previously given.

    # PLCC: 0.2386 SROCC: 0.2524 KROCC: 0.1637 BRISQUE Trained on ISIQA
    # PLCC: 0.3737 SROCC: 0.3854 KROCC: 0.2469 DIIVINE Trained on ISIQA

    # PLCC: 0.0406 SROCC: 0.0661 KROCC: 0.0428 BRISQUE scores
    # PLCC: 0.1024 SROCC: 0.1403 KROCC: 0.0922 NIQE scores
    # PLCC: 0.0651 SROCC: 0.1091 KROCC: 0.0664 DIIVINE

