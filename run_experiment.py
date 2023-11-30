import numpy as np
import single_trial
import models
import odors
from odors import odor_input
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def run_model1(trials_FOC, trials_SOC, num_KC, KC_baseline, odor_activation, odor_FOC, odor_SOC, init_KCMBONp,
            init_KCMBONn, initMBONpDAN, LR_KCMBONn, LR_MBONpDAN, DAN_activation=False):
    '''

       runs the network simulation for any number of trials for N model instance

       :param trials_FOC: (int) number of first order conditioning trials
       :param trials_SOC: (int) number of second order conditioning trials
       :param num_KC: (int) number of KC neurons
       :param KC_baseline: (num) KC baseline spike rate
       :param odor_activation: (num) KC odor activation rate
       :param odor_FOC: (iterable) odor activation pattern
       :param odor_SOC: (iterable) odor activation pattern
       :param init_KCMBONp: (num) initialization weights wKC_MBONp
       :param init_KCMBONn: (num) initialization weights wKC_MBONn
       :param initMBONpDAN: (num) initialization weights wMBONp_DAN
       :param LR_KCMBONp: (num) learning rate
       :param DAN_activation: (num) DAN spike rate

       :return: DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONp_DAN_weights
                np.arrays with trials + 1 elements/columns, the first element/column contains the initial value

       '''


    trials = trials_FOC + trials_SOC

    # save DAN , KC, MBON rates and all weights for each trial
    DAN_rate = np.zeros(trials + 1)
    MBONp_rate = np.zeros(trials + 1)
    MBONn_rate = np.zeros(trials + 1)

    KC_MBONp_weights = np.zeros((num_KC, trials + 1))
    KC_MBONn_weights = np.zeros((num_KC, trials + 1))
    MBONp_DAN_weights = np.zeros(trials + 1)



    # create and initialize network
    KC, wKC_MBONp, wKC_MBONn, wMBONp_DAN, MBONp, MBONn, DAN = models.create_model1(num_KC=num_KC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_MBONpDAN=initMBONpDAN)

    # save initial values
    DAN_rate[0] = DAN
    MBONp_rate[0] = MBONp
    MBONn_rate[0] = MBONn
    KC_MBONp_weights[:, 0] = wKC_MBONp
    KC_MBONn_weights[:, 0] = wKC_MBONn
    MBONp_DAN_weights[0] = wMBONp_DAN






    # execute experiment

    for i, trial in enumerate(np.arange(1, trials_FOC + 1)):

        KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wMBONp_DAN = single_trial.single_trial_model1(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation, odor=odor_FOC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_MBONpDAN=initMBONpDAN, KC=KC,wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, wMBONp_DAN=wMBONp_DAN, DAN=DAN, LR_KCMBONn=LR_KCMBONn,LR_MBONpDAN=LR_MBONpDAN, DAN_activation=DAN_activation)


        # save values
        DAN_rate[trial] = DAN
        MBONp_rate[trial] = MBONp
        MBONn_rate[trial] = MBONn
        KC_MBONp_weights[:, trial] = wKC_MBONp
        KC_MBONn_weights[:, trial] = wKC_MBONn
        MBONp_DAN_weights[trial] = wMBONp_DAN


    for i, trial in enumerate(np.arange(trials_SOC + 1, trials + 1)):
        KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wMBONp_DAN = single_trial.single_trial_model1(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation, odor=odor_SOC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_MBONpDAN=initMBONpDAN, KC=KC,wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, wMBONp_DAN=wMBONp_DAN, DAN=DAN, LR_KCMBONn=LR_KCMBONn, LR_MBONpDAN=LR_MBONpDAN, DAN_activation=0)

        # save values
        DAN_rate[trial] = DAN
        MBONp_rate[trial] = MBONp
        MBONn_rate[trial] = MBONn
        KC_MBONp_weights[:, trial] = wKC_MBONp
        KC_MBONn_weights[:, trial] = wKC_MBONn
        MBONp_DAN_weights[trial] = wMBONp_DAN

    return DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONp_DAN_weights


DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONp_DAN_weights = run_model1(trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0, odor_activation=5, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.09,
            init_KCMBONn=0.09, initMBONpDAN=0 , LR_KCMBONn=0.0012, LR_MBONpDAN=0.001,DAN_activation=15)





