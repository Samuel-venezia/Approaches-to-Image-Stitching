import pandas as pd
from assets import stitched_approaches
import os
from Dataset_Panorama import stitched_final


def stitching_pan(ds_name, col_app, directory_names, dur, app, fb_pipeline=""):
    for dir_ind, dir_name in enumerate(directory_names):
        output_dir = f"output_stitched/{ds_name}/{dir_name}"
        if dir_name == "office":
            pass
            dir_reverse = False
        else:
            if dir_ind % 2 == 0:
                dir_reverse = False
            else:
                dir_reverse = True
        duration_approach = stitched_final(dir_reverse=dir_reverse, dir_name=f"{ds_name}/{dir_name}",
                                           output_dir=output_dir, approach=app,
                                           width=1024, pipeline_stages=fb_pipeline)
        dur.append(duration_approach)
        if len(fb_pipeline) > 0:
            col_app.append(f"{app}-{fb_pipeline}")
        else:
            col_app.append(f"{app}")
        # duration approach can be appended into a list,
        # and used within a dataframe as a comparison with other approaches
        print(f"Duration of {app} for {dir_name}: {duration_approach} seconds")


if __name__ == "__main__":
    stitched_pipelines = ["AKAZE-SIFT-BF KNN-RANSAC", "AKAZE-FREAK-BF-RANSAC", "AKAZE-BRISK-BF-RANSAC",
                          "AKAZE-SIFT-BF KNN-USAC", "AKAZE-SIFT-BF-USAC", "SIFT-SIFT-BF KNN-USAC"]

    dataset_name = "adobe_panoramas"
    dir_names = os.listdir(dataset_name)
    durations = []
    col_app = []
    scene_names = []
    for approach in stitched_approaches:
        scene_names.extend(dir_names)
        if approach == stitched_approaches[0]:
            for pipeline in stitched_pipelines:
                stitching_pan(dataset_name, col_app, dir_names, durations, approach, pipeline)
        else:
            stitching_pan(dataset_name, col_app, dir_names, durations, approach)

    duration_df = pd.DataFrame([col_app, scene_names, durations], columns=["Approach", "Scene Name", "Duration"])
    print(duration_df.to_string())
    # duration_df.to_csv(f"output_stitched/{dataset_name}/durations.csv")
