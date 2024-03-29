import numpy as np
import single_trial
import models
import models_var_connectivity
import models_var_weights
import preference_test
import matplotlib as mpl
import matplotlib.pyplot as plt


def run(seed_param,trials_FOC, trials_SOC, num_KC, KC_baseline, odor_activation, odor_FOC, odor_SOC, init_KCMBONp,
            init_KCMBONn, initMBONnDAN, LR_KCMBONn, DANbaseline, DAN_activation=False,variable_connectivity=False,variable_weight=False,generalization_exp=False):
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
       :param initMBONnDAN: (num) initialization weights wMBONn_DAN
       :param LR_KCMBONn: (num) learning rate
       :param DANbaseline: (num) DAN spike rate at baseline (no odor)
       :param DAN_activation: (num) DAN spike rate
       :param variable_connectivity:

       :return: DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONn_DAN_weights, test_results

       np.arrays with shape(number of neurons/weights, trials + 1 element/column),
       the first element/column contains the initial value
       '''

    trials = trials_FOC + trials_SOC

    if generalization_exp == True:
        trials = trials_FOC + trials_SOC + 3

    # save DAN , KC, MBON rates and all weights for each trial
    DAN_rate = np.zeros(trials + 1)
    MBONp_rate = np.zeros(trials + 1)
    MBONn_rate = np.zeros(trials + 1)
    KC_MBONp_weights = np.zeros((num_KC, trials + 1))
    KC_MBONn_weights = np.zeros((num_KC, trials + 1))
    MBONp_DAN_weights = np.zeros(trials + 1)
    MBONn_DAN_weights = np.zeros(trials + 1)

    test_results = dict()

    # create and initialize network
    KC, wKC_MBONp, wKC_MBONn, wMBONn_DAN, MBONp, MBONn, DAN,DAN_baseline = models.create_model4(num_KC=num_KC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_MBONnDAN=initMBONnDAN, DAN_baseline=DANbaseline)

    if variable_connectivity == True:
        KC, wKC_MBONp, wKC_MBONn, wMBONn_DAN, MBONp, MBONn, DAN, DAN_baseline = models_var_connectivity.create_model4(num_KC=num_KC,
                                                                                                     init_KCMBONp=init_KCMBONp,
                                                                                                     init_KCMBONn=init_KCMBONn,
                                                                                                     init_MBONnDAN=initMBONnDAN,
                                                                                                     DAN_baseline=DANbaseline)

    if variable_weight == True:
        KC, wKC_MBONp, wKC_MBONn, wMBONn_DAN, MBONp, MBONn, DAN, DAN_baseline = models_var_connectivity.create_model4(
            num_KC=num_KC,
            init_KCMBONp=init_KCMBONp,
            init_KCMBONn=init_KCMBONn,
            init_MBONnDAN=initMBONnDAN,
            DAN_baseline=DANbaseline)

    # save initial values
    DAN_rate[0] = DAN
    MBONp_rate[0] = MBONp
    MBONn_rate[0] = MBONn
    KC_MBONp_weights[:, 0] = wKC_MBONp
    KC_MBONn_weights[:, 0] = wKC_MBONn
    MBONn_DAN_weights[0] = wMBONn_DAN

    # execute experiment
    for i, trial in enumerate(np.arange(1, trials_FOC + 1)):
        KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wMBONn_DAN = single_trial.single_trial_model4(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation, odor=odor_FOC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_MBONnDAN=initMBONnDAN, KC=KC,wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, wMBONn_DAN=wMBONn_DAN, DAN=DAN, LR_KCMBONn=LR_KCMBONn,DAN_activation=DAN_activation, DAN_baseline=DANbaseline,ref_odor=None,seed_param=seed_param)

        # save values
        DAN_rate[trial] = DAN
        MBONp_rate[trial] = MBONp
        MBONn_rate[trial] = MBONn
        KC_MBONp_weights[:, trial] = wKC_MBONp
        KC_MBONn_weights[:, trial] = wKC_MBONn
        MBONn_DAN_weights[trial] = wMBONn_DAN

    pref_odor1 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor1', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,seed_param=seed_param)
    test_results.update({"post FOC odor1": pref_odor1})
    pref_odor2 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor2', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,seed_param=seed_param)
    test_results.update({"post FOC odor2": pref_odor2})
    pref_odor3 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor3', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,seed_param=seed_param)
    test_results.update({"post FOC odor3": pref_odor3})

    for i, trial in enumerate(np.arange(trials_SOC + 1, trials + 1)):
        KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wMBONp_DAN = single_trial.single_trial_model4(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation, odor=odor_SOC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_MBONnDAN=initMBONnDAN, KC=KC,wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,  wMBONn_DAN=wMBONn_DAN, DAN=DAN, LR_KCMBONn=LR_KCMBONn, DAN_activation=0, DAN_baseline=DANbaseline,ref_odor=None,seed_param=seed_param)

        # save values
        DAN_rate[trial] = DAN
        MBONp_rate[trial] = MBONp
        MBONn_rate[trial] = MBONn
        KC_MBONp_weights[:, trial] = wKC_MBONp
        KC_MBONn_weights[:, trial] = wKC_MBONn
        MBONn_DAN_weights[trial] = wMBONn_DAN

    pref_odor1 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor1', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,seed_param=seed_param)
    test_results.update({"post SOC odor1": pref_odor1})
    pref_odor2 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor2', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,seed_param=seed_param)
    test_results.update({"post SOC odor2": pref_odor2})
    pref_odor3 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor3', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,seed_param=seed_param)
    test_results.update({"post SOC odor3": pref_odor3})

    if generalization_exp==True:
        for i, trial in enumerate(np.arange(3)):
            KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wMBONp_DAN = single_trial.single_trial_model4(num_KC=num_KC,
                                                                                                       KC_baseline=KC_baseline,
                                                                                                       odor_activation=odor_activation,
                                                                                                       odor='odor3',
                                                                                                       init_KCMBONp=init_KCMBONp,
                                                                                                       init_KCMBONn=init_KCMBONn,
                                                                                                       init_MBONnDAN=initMBONnDAN,
                                                                                                       KC=KC,
                                                                                                       wKC_MBONp=wKC_MBONp,
                                                                                                       wKC_MBONn=wKC_MBONn,
                                                                                                       wMBONn_DAN=wMBONn_DAN,
                                                                                                       DAN=DAN,
                                                                                                       LR_KCMBONn=LR_KCMBONn,
                                                                                                       DAN_activation=0,
                                                                                                       DAN_baseline=DANbaseline,
                                                                                                       ref_odor=None,
                                                                                                       seed_param=seed_param)
        pref_odor1 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline,
                                                 odor_activation=odor_activation, test_odor='odor1',
                                                 wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, seed_param=seed_param)
        test_results.update({"post generalization odor1": pref_odor1})
        pref_odor2 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline,
                                                 odor_activation=odor_activation, test_odor='odor2',
                                                 wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, seed_param=seed_param)
        test_results.update({"post generalization odor2": pref_odor2})
        pref_odor3 = preference_test.test_model1(num_KC=num_KC, KC_baseline=KC_baseline,
                                                 odor_activation=odor_activation, test_odor='odor3',
                                                 wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, seed_param=seed_param)
        test_results.update({"post generalization odor3": pref_odor3})

    return DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONn_DAN_weights, test_results



if __name__ == '__main__':
    DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights,  MBONn_DAN_weights, test_results = run(seed_param=np.random.randint(1,999,1),trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0, odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
             init_KCMBONn=0.083, initMBONnDAN=0.131313, LR_KCMBONn=0.003182, DANbaseline=11.212121, DAN_activation=3.727273,generalization_exp=True)


