import numpy as np
import os
from matplotlib import pyplot as plt
import run_model1
import run_model2
import run_model3
import run_model4
import run_model5
from tqdm import tqdm
#import pandas as pd
# argument parser for model selection
import sys

def sample_hypersphere(dims=1, samples=1, radius=1, x_base=None):
    '''
    Sample the surface of a hypersphere of dimension dims with a radius around the origin x_base
    to test how many points lie in the hypercube [0, 1]^4 with inreasing radius of the hypersphere

    :param dims: dimensions of the hypershpere = number of optimised model parameters (num)
    :param samples: number of samples drawn from the hypersphere surface (num)
    :param radius: (num)
    :param x_base: initial position of the hypersphere with radius = 0  (np.array)

    :return: points on the hypersphere surface
    '''

    if x_base is None:
        x_base=np.zeros(dims)
    points=np.random.randn(samples, dims)

    return radius * points / np.linalg.norm(points, axis=1)[:, None]+x_base



# Evaluation function of model with parameters and test if result is not optimal anymore
def evaluate(points, f, f_optimal, base_parameter=np.nan, scale_parameter=np.nan):
    '''

    Evaluate the function f which is our model simulation of the points
    f_optimal should return in our case 0 or 1 -> It's an indicator function
    At some point the parameters have to be scaled to the original values instead of the [0,1] range and the

    :param points: sampled points from the hypersphere surface
    :param f: model simulation function
    :param f_optimal: indicator function for optimal solution
    :param base_parameter: initial position of the hypersphere
    :param scale_parameter: parameter for standardisation

    :return:
    '''

    if np.isnan(base_parameter).any():
        base_parameter=np.zeros(len(points[0]))
    if np.isnan(scale_parameter).any():
        scale_parameter=np.ones(len(points[0]))

    # scale points to original values
    points = points * scale_parameter + base_parameter  # transforms z-scored points to original parameters
    # apply f to every point in points
    result= [f(point) for point in points]
    #df = pd.DataFrame(result)
    #append the points to the df in 4 columns
    # df['point1'] = points[:,0]
    # df['point2'] = points[:,1]
    # df['point3'] = points[:,2]
    # df['point4'] = points[:,3]


    result = [f_optimal(point) for point in result] # list of ones and zeros of len(points)

    return np.mean(result) # yields the % of optimal points in the sample if multiplied with 100

def f_optimal(test_results,model_fct):
    '''
    Indicator function for optimal solution

    :param test_results: parameter combination to be evaluated
    :param model_fct: run function of  the model to be evaluated

    :return: parameter combination yield optimal results or not (bool)
    '''


    # test for rate conditions
    if any(test_results['DAN_rate'] > 15) or any(test_results['MBONp_rate'] > 60) or any(test_results['MBONn_rate'] > 60) or any(test_results['DAN_rate'] < 0) or any(test_results['MBONp_rate'] < 0) or any(test_results['MBONn_rate'] < 0):
        return 0

    if model_fct == run_model1:
        if (test_results['post FOC odor1'] == 1 and test_results['post FOC odor2'] == 0 and test_results[
            'post FOC odor3'] == 0 and test_results['post SOC odor1'] == 1 and test_results[
            'post SOC odor2'] >= 0.0221 and test_results['post SOC odor3'] == 0):
            return 1  # the parameter combination in point results fulfills all criteria for the optimal learner
        else:
            return 0

    else:
        if (test_results['post FOC odor1'] == 1 and test_results['post FOC odor2'] == 0 and test_results[
            'post FOC odor3'] == 0 and test_results['post SOC odor1'] == 1 and test_results[
            'post SOC odor2'] >= 0.333 and test_results['post SOC odor3'] == 0):
            return 1 # the parameter combination in point results fulfills all criteria for the optimal learner
        else:
            return 0

def f(point, model_fct):

    '''
    Runs the model simulation with the parameter set to be tested

    :param point: parameter combination
    :param model_fct: model to test

    :return: model learning test results (dict)
    '''


    if model_fct == run_model1:
        DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, KC_KC_weights, KC_DAN_weights, test_results = model_fct.run(
            seed_param=np.random.randint(1, 999, 1), trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0,
            odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
            init_KCMBONn=0.083, init_KCKC=0, init_KCDAN=point[0], LR_KCMBONn=point[1], LR_KCKC=point[2],
            DAN_activation=point[3])


    if model_fct == run_model2:
        DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, KC_DAN_weights, test_results = model_fct.run(
            seed_param=np.random.randint(1, 999, 1), trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0,
            odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
            init_KCMBONn=0.083, initKCDAN=point[0], LR_KCMBONn=point[1], LR_KCDAN=point[2], DAN_activation=point[3])

    if model_fct == run_model3:
        DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONp_DAN_weights, MBONn_DAN_weights, test_results = model_fct.run(
            seed_param=np.random.randint(1, 999, 1), trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0,
            odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
            init_KCMBONn=0.083, initMBONpDAN=point[0], initMBONnDAN=point[1], LR_KCMBONn=point[2], DAN_activation=point[3])

    if model_fct == run_model4:
        DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONn_DAN_weights, test_results = model_fct.run(
            seed_param=np.random.randint(1, 999, 1), trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0,
            odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
            init_KCMBONn=0.083, initMBONnDAN=point[0], LR_KCMBONn=point[1], DANbaseline=point[2],
            DAN_activation=point[3])

    if model_fct == run_model5:
        DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONp_DAN_weights, test_results = model_fct.run(
            seed_param=np.random.randint(1, 999, 1), trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0,
            odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
            init_KCMBONn=0.083, initMBONpDAN=point[0], LR_KCMBONn=point[1], LR_MBONpDAN=point[2],
            DAN_activation=point[3])

    rate_dict = {'DAN_rate': DAN_rate, 'MBONp_rate': MBONp_rate, 'MBONn_rate': MBONn_rate}
    test_results.update(rate_dict)

    return test_results


if __name__ == '__main__':
    model = 5 # specify model
    # model = int(sys.argv[1]) if len(sys.argv) > 1 else 1 # add commond line argument for model selection with default value 1
    x = 1  # multiply the scale parameters for the maximum to extend the hypercube tested
    SampleSize = 700
    n_radii = 100
    Dimensions = 4
    radius = np.linspace(start=0.0,stop=1.0,num=n_radii) # times the scale parameter it results in a hypercube with a dimater of 2
    base_model1 = [0.001, 0.004, 0.000162, 7.272727]
    base_model2 = [0.000505, 0.003333, 0.000677, 5.727273]
    base_model3 = [0.272727, 0.262626, 0.003121, 5.0]
    base_model4 = [0.131313, 0.003182, 11.212121, 3.727273]
    base_model5 = [0.08008, 0.003182, 0.003, 4.727273]

    base_model= [base_model1, base_model2, base_model3, base_model4, base_model5]

    scale_param1 = np.array([0.001 - 0, 0.004 - 0.001, 0.004 - 0, 10 - 1])
    scale_param2 = np.array([0.001-0,0.004-0.001,0.001-0,10-1])
    scale_param3 = np.array([1-0,1-0,0.004-0.001,10-1])
    scale_param4 = np.array([0.5-0,0.004-0.001,30-0,10-1])
    scale_param5 = np.array([0.4 - 0, 0.004 - 0.001, 0.01 - 0.001, 10 - 1])


    scale= [scale_param1, scale_param2, scale_param3, scale_param4, scale_param5]

    model_function = [run_model1, run_model2, run_model3, run_model4, run_model5]

    base_parameter = np.array(base_model[model-1]) # selected central optimal point for each model
    model_fct = model_function[model-1] # indicates the model to be evaluated

    scale_param = x*scale[model-1] # one for each model
    filename = 'model'+str(model) # filename for saving data to file

    optimal=np.zeros(len(radius))
    for i in tqdm(range(len(radius))):
        points=sample_hypersphere(Dimensions, SampleSize, radius[i]) # sample points around origin, don't apply base_parameter here
        # they would be also scaled with the scale_parameter afterwards
        optimal[i]=evaluate(points, lambda point:f(point, model_fct), lambda test_results:f_optimal(test_results, model_fct), base_parameter,scale_parameter=scale_param)




