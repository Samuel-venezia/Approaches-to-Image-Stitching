import numpy as np
from brisque import BRISQUE
import pandas as pd
import pyiqa
import cv2 as cv
import matplotlib.pyplot as plt
from ComputeSIQE import corr_coeff
import random
from sklearn import svm
import joblib
from pathlib import Path


def diivine(mos):
    trained = True
    if not trained:
        diivine_df = pd.read_csv(f"../Comb_SIQE/isiqa_divine_scores.csv")
        diivine_scores = [np.float32(f) for f in diivine_df.columns.tolist()]
    else:
        svm_reg = joblib.load(f"diivine_model.pkl")
        features = getting_features("diivine")
        diivine_scores = svm_reg.predict(features).tolist()
    plcc, srocc, krocc = corr_coeff(diivine_scores, mos)
    plt.scatter(mos, diivine_scores, marker="x", color="black")
    plt.xlabel("MOS")
    plt.ylabel("DIIVINE")
    plt.xlim(left=0, right=100)
    plt.ylim(bottom=0, top=100)
    plt.savefig("SP_DIIVINE_trained_ISIQA.png")


def niqe(mos):
    niqe_test = pyiqa.create_metric("niqe")
    scores = []
    for index in range(264):
        scores.append(niqe_test(f"../isiqa_release/stitched_images/{index + 1}.jpg").item())
        print(f"NIQE calculated: {index + 1}")
    plcc, srocc, krocc = corr_coeff(scores, mos)
    plt.scatter(mos, scores, marker="x", color="black")
    plt.xlabel("MOS")
    plt.ylabel("NIQE")
    plt.xlim(left=0, right=100)
    plt.ylim(bottom=0)
    plt.savefig("SP_NIQE_ISIQA.png")


def brisque(mos):
    trained = True
    if not trained:
        brisque = BRISQUE()
        scores = []
        for index in range(264):
            scores.append(float(brisque.score(cv.imread(f"../isiqa_release/stitched_images/{index + 1}.jpg"))))
            print(f"BRISQUE calculated: {index + 1}")
    else:
        svm_reg = joblib.load(f"brisque_model.pkl")
        features = getting_features("brisque")
        scores = svm_reg.predict(features).tolist()
    plcc, srocc, krocc = corr_coeff(scores, mos)
    plt.scatter(mos, scores, marker="x", color="black")
    plt.xlabel("MOS")
    plt.ylabel("BRISQUE")
    plt.xlim(left=0, right=100)
    plt.ylim(bottom=0, top=100)
    plt.savefig("SP_BRISQUE_trained_ISIQA.png")


def combsiqe_modified(mos):
    svm_reg = joblib.load(f"COMBSIQE_model.pkl")
    features = combsiqe_values()
    scores = svm_reg.predict(features).tolist()
    plcc, srocc, krocc = corr_coeff(scores, mos)
    plt.scatter(mos, scores, marker="x", color="black")
    plt.xlabel("MOS")
    plt.ylabel("COMBSIQE")
    plt.xlim(left=0, right=100)
    plt.ylim(bottom=0, top=100)
    plt.savefig("SP_COMBSIQE_svm_trained_ISIQA.png")


def getting_features(iqa_type="brisque"):
    # returns a list of features
    feature_df = pd.read_csv(f"../Comb_SIQE/isiqa_{iqa_type}_features.csv", header=None)
    features = feature_df.values.tolist()[0]
    feat_images = []
    for feat in range(0, len(features), int(len(features)/264)):
        feat_images.append([np.float32(f) for f in features[feat:feat+int(len(features)/264)]])
    return feat_images


def training_testing_mos(no_scenes=5):
    isiqa_breakdown = [9, 12, 12, 11, 11, 11, 12, 8, 10, 7, 9, 10, 9, 11, 10, 6, 9, 10, 12, 12, 13, 8, 9, 12, 12, 9]
    # Split training/testing
    # returns the MOS for training a SVM and the index list for the inputs
    # retrieving the output MOS for the testing subset
    # Select 21 scenes from the dataset, or just select 5 scenes.
    # RNG from a list, remove value when selected.
    mos_file = open("../isiqa_release/MOS.txt", "r")
    mean_opinion_scores = [float(line) for line in mos_file.readlines()]
    test_scenes = random.sample(range(1,27), no_scenes)
    train_scenes = [scene for scene in range(1, 27)]
    train_scenes = [train for train in train_scenes if train not in test_scenes]
    train_output_index = []
    for train_val in train_scenes:
        train_img_start = sum(isiqa_breakdown[:train_val-1])
        train_img_end = sum(isiqa_breakdown[:train_val])
        train_output_index.extend([out_index for out_index in range(train_img_start, train_img_end)])
    return [mean_opinion_scores[i] for i in train_output_index], train_output_index


def training_svm(type="brisque"):
    for i in range(1000):
        print(f"Iteration {i+1} of 1000")
        if Path(f"{type}_model.pkl").is_file():
            # load
            svm_reg = joblib.load(f"{type}_model.pkl")
        else:
            svm_reg = svm.SVR()
        if "COMBSIQE" in type:
            features = combsiqe_values()
        else:
            features = getting_features(type)
        outputs, features_index = training_testing_mos()
        input_features = [features[i] for i in features_index]
        svm_reg.fit(input_features, outputs)
        # save
        joblib.dump(svm_reg, f"{type}_model.pkl")
    print("No errors and saved")


def combsiqe_values():
    features_list = []
    for subset_index in range(1,27):
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
        features_subset = pd.DataFrame(data=[f1s,f2s,f3s,f4s,f5s,f6s,f7s,f8s]).transpose()
        features_subset.columns = ["F1","F2","F3","F4","F5","F6","F7","F8"]
        features_list.append(features_subset)
    features_df = pd.concat(features_list, ignore_index=True)
    inputs = []
    for img_index in range(264):
        inputs.append(features_df.iloc[img_index].tolist())
    return inputs


if __name__ == "__main__":
    mos_file = open("../isiqa_release/MOS.txt", "r")
    mean_opinion_scores = [float(line) for line in mos_file.readlines()]
    mos_file.close()
    # training_svm("COMBSIQE")
    combsiqe_modified(mean_opinion_scores)
    diivine(mos=mean_opinion_scores)
    brisque(mean_opinion_scores)
    # PLCC: 0.2386 SROCC: 0.2524 KROCC: 0.1637 BRISQUE TRAINED
    # PLCC: 0.3737 SROCC: 0.3854 KROCC: 0.2469 DIIVINE TRAINED
    # PLCC: 0.7449 SROCC: 0.6988 KROCC: 0.5260 Modified COMBSIQE TRAINED
    # Metrics trained using their default datasets
    # PLCC: 0.0406 SROCC: 0.0661 KROCC: 0.0428 BRISQUE
    # PLCC: 0.1024 SROCC: 0.1403 KROCC: 0.0922 NIQE
    # PLCC: 0.0651 SROCC: 0.1091 KROCC: 0.0664 DIIVINE
