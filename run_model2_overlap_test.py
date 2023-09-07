import numpy as np
import single_trial
import models
import preference_test
import matplotlib as mpl
import matplotlib.pyplot as plt



def run(seed_param,trials_FOC, trials_SOC, num_KC, KC_baseline, odor_activation, odor_FOC, odor_SOC, init_KCMBONp,
            init_KCMBONn, initKCDAN, LR_KCMBONn, LR_KCDAN, overlap,ref_odor, DAN_activation=False):
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
       :param initKCDAN: (num) initialization weights wMBONp_DAN
       :param LR_KCMBONn: (num) learning rate
       :param LR_KCDAN: (num) learning rate
       :param DAN_activation: (num) DAN spike rate

       :return: DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, KC_DAN_weights, test_results

       np.arrays with shape(number of neurons/weights, trials + 1 element/column),
       the first element/column contains the initial value

       '''

    trials = trials_FOC + trials_SOC

    # save DAN , KC, MBON rates and all weights for each trial
    DAN_rate = np.zeros(trials + 1)
    MBONp_rate = np.zeros(trials + 1)
    MBONn_rate = np.zeros(trials + 1)
    KC_MBONp_weights = np.zeros((num_KC, trials + 1))
    KC_MBONn_weights = np.zeros((num_KC, trials + 1))
    KC_DAN_weights = np.zeros((num_KC,trials + 1))

    test_results = dict()

    # create and initialize network
    KC, wKC_MBONp, wKC_MBONn, wKC_DAN, MBONp, MBONn, DAN = models.create_model2(num_KC=num_KC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_KCDAN=initKCDAN)

    # save initial values
    DAN_rate[0] = DAN
    MBONp_rate[0] = MBONp
    MBONn_rate[0] = MBONn
    KC_MBONp_weights[:, 0] = wKC_MBONp
    KC_MBONn_weights[:, 0] = wKC_MBONn
    KC_DAN_weights[:,0] = wKC_DAN

    # execute experiment
    for i, trial in enumerate(np.arange(1, trials_FOC + 1)):
        KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wKC_DAN = single_trial.single_trial_model2(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation, odor=odor_FOC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_KCDAN=initKCDAN, KC=KC,wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, wKC_DAN=wKC_DAN, DAN=DAN, LR_KCMBONn=LR_KCMBONn,LR_KCDAN=LR_KCDAN, DAN_activation=DAN_activation,seed_param=seed_param,ref_odor=ref_odor,overlap=overlap,overlap_on=True)

        # save values
        DAN_rate[trial] = DAN
        MBONp_rate[trial] = MBONp
        MBONn_rate[trial] = MBONn
        KC_MBONp_weights[:, trial] = wKC_MBONp
        KC_MBONn_weights[:, trial] = wKC_MBONn
        KC_DAN_weights[:,trial] = wKC_DAN

    pref_odor1 = preference_test.test_model_overlap_test(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor1', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,overlap=overlap,ref_odor=ref_odor,seed_param=seed_param)
    test_results.update({"post FOC odor1": pref_odor1})
    pref_odor2 = preference_test.test_model_overlap_test(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor2', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,overlap=overlap,ref_odor=ref_odor,seed_param=seed_param)
    test_results.update({"post FOC odor2": pref_odor2})
    pref_odor3 = preference_test.test_model_overlap_test(num_KC=num_KC, KC_baseline=KC_baseline,
                                                    odor_activation=odor_activation, test_odor='odor3',
                                                    wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, overlap=overlap,
                                                    ref_odor=ref_odor, seed_param=seed_param)
    test_results.update({"post FOC odor3": pref_odor3})


    for i, trial in enumerate(np.arange(trials_SOC + 1, trials + 1)):
        KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wKC_DAN = single_trial.single_trial_model2(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation, odor=odor_SOC, init_KCMBONp=init_KCMBONp, init_KCMBONn=init_KCMBONn, init_KCDAN=initKCDAN, KC=KC,wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, wKC_DAN=wKC_DAN, DAN=DAN, LR_KCMBONn=LR_KCMBONn, LR_KCDAN=LR_KCDAN, DAN_activation=0,seed_param=seed_param,ref_odor=ref_odor,overlap=overlap,overlap_on=True)

        # save values
        DAN_rate[trial] = DAN
        MBONp_rate[trial] = MBONp
        MBONn_rate[trial] = MBONn
        KC_MBONp_weights[:, trial] = wKC_MBONp
        KC_MBONn_weights[:, trial] = wKC_MBONn
        KC_DAN_weights[:,trial] = wKC_DAN

    pref_odor1 = preference_test.test_model_overlap_test(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor1', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,overlap=overlap,ref_odor=ref_odor,seed_param=seed_param)
    test_results.update({"post SOC odor1": pref_odor1})
    pref_odor2 = preference_test.test_model_overlap_test(num_KC=num_KC, KC_baseline=KC_baseline, odor_activation=odor_activation,test_odor='odor2', wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn,overlap=overlap,ref_odor=ref_odor,seed_param=seed_param)
    test_results.update({"post SOC odor2": pref_odor2})
    pref_odor3 = preference_test.test_model_overlap_test(num_KC=num_KC, KC_baseline=KC_baseline,
                                                    odor_activation=odor_activation, test_odor='odor3',
                                                    wKC_MBONp=wKC_MBONp, wKC_MBONn=wKC_MBONn, overlap=overlap,
                                                    ref_odor=ref_odor, seed_param=seed_param)
    test_results.update({"post SOC odor3": pref_odor3})

    return DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, KC_DAN_weights, test_results




