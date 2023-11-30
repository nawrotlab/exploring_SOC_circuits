import numpy as np
import pandas as pd
import scipy.stats
from pandarallel import pandarallel
import os
from scipy.spatial.distance import cdist


def find_optimal_model(path,data_frame,parameters):
    '''

    :param path:
    :param data_frame:
    :param parameters:

    :return: parameters of the optimal model
    '''

    df_model = pd.read_csv(path + data_frame) # read data frame
    df_model = df_model[parameters] # clean data frame

    # find average data point
    average_point = scipy.stats.zscore(df_model.mean(axis=0))

    # compute all distances between the average point and all rows
    df_model.apply(scipy.stats.zscore)
    df_model['distance'] = (cdist([average_point], df_model.values)[0])
    optimal_model = df_model.iloc[df_model['distance'].idxmin()]


    return optimal_model






#
# def find_optimal_model(path, data_frame, parameters):
#     '''
#
#     :param data_frame:
#     :return:
#     '''
#
#     CPUs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
#     pandarallel.initialize(nb_workers=CPUs, progress_bar=True)
#
#     JobID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
#
#     NumJobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 100))
#
#
#
#
#     df_model = pd.read_csv(path + data_frame)
#     len(df_model)
#     df_model = df_model[parameters] # clean data frage so it only consists of the parameters
#
#     # compute euclidean distances between all rows
#     df_model.apply(scipy.stats.zscore) # standardizing data to a mean=0, std=1
#     df_model_subset = df_model.iloc[]
#     df_model_subset.parallel_apply(lambda row: np.sum([np.linalg.norm(row.values - df_model.loc[[_id], :].values, 2) for _id in df_model.index.values]), axis=1)
#
#
#     return