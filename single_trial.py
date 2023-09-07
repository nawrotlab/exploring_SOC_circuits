import matplotlib.pyplot as plt
import numpy as np
import models
import odors
from odors import odor_input_overlap


def single_trial_model1(num_KC, KC_baseline, odor_activation, odor, init_KCMBONp, init_KCMBONn, init_KCDAN,init_KCKC, KC, wKC_MBONp, wKC_MBONn,wKC_DAN, wKC_KC, DAN, LR_KCMBONn, LR_KCKC, DAN_activation,seed_param,ref_odor=None,overlap=0,overlap_on=False):
    '''
        simulates a single trial

        :param num_KC: (int) number of KC neurons
        :param KC_baseline: (num) KC baseline spike rate
        :param odor_activation: (num) KC odor activation rate
        :param odor: (iterable) odor activation pattern
        :param init_KCMBONp: (num) initialization weights wKC_MBONp
        :param init_KCMBONn: (num) initialization weights wKC_MBONn
        :param init_MBONpDAN: (num) initialization weights wMBONpDAN
        :param KC: (np.array) KC activation pattern
        :param wKC_MBONp: (np.array) synaptic weight matrix
        :param wKC_MBONn: (np.array) synaptic weight matrix
        :param wMBONp_DAN: (np.array) synaptic weight matrix
        :param MBONp: (num) MBON activation
        :param MBONn: (num) MBON activation
        :param DAN: (num) DAN activation
        :param LR_KCMBONn: (num) learning rate
        :param DAN_activation: (num) DAN spike rate
        :param seed_param: (num) numpy seed for odor generation

        :return:
        KC (np.array) KC spike rates
        DAN (np.array) DAN spike rates
        MBONp (np.array) MBON spike rates
        MBONn (np.array) MBON spike rates
        weights (dict) all synaptic weights

        '''

    # compute KC odor activation

    if ref_odor == 'odor1':
        experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                   overlap=overlap, ref_odor=ref_odor, seed_param=seed_param)

    elif ref_odor == 'odor2':
        experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                   overlap=overlap, ref_odor=ref_odor, seed_param=seed_param)

    elif overlap_on == True:
        experiment_odors = odors.odor_input_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                    overlap=overlap, seed_param=seed_param)

    else:
        experiment_odors = odors.odor_input(KC_baseline, num_KC, odor_activation, seed_param)


    KC = experiment_odors[odor]

    # compute MBON output
    MBONp = np.dot(KC, wKC_MBONp)
    MBONn = np.dot(KC, wKC_MBONn)


    # compute DAN activation
    DAN = DAN_activation + np.dot(KC, wKC_DAN)
    if DAN < 0:
        DAN = 0

    # plasticity wKC_MBONn (coincidence of odor and reinforcement triggers plasticity)
    # and  plasticity KC-KC (coincidence of KC-KC activation triggers increase of the w)
    active_KC = np.where(KC > KC_baseline)[0]

    for neuron in active_KC:

        wKC_KC[neuron, active_KC] += LR_KCKC

        if wKC_MBONn[neuron] > (0 + LR_KCMBONn * DAN):  # ensure that weight never become negative
            wKC_MBONn[neuron] -= LR_KCMBONn * DAN

        if wKC_MBONn[neuron] <= (0 + LR_KCMBONn * DAN):
            wKC_MBONn[neuron] = 0

    np.fill_diagonal(wKC_KC, 0)  # weights of KC synapses with themselves are set to 0



    return KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wKC_DAN, wKC_KC


def single_trial_model2(num_KC, KC_baseline, odor_activation, odor, init_KCMBONp, init_KCMBONn, init_KCDAN, KC, wKC_MBONp, wKC_MBONn, wKC_DAN, DAN, LR_KCMBONn,LR_KCDAN, DAN_activation,seed_param,ref_odor=None,overlap=0,overlap_on=False):
    '''
        simulates a single trial

        :param num_KC: (int) number of KC neurons
        :param KC_baseline: (num) KC baseline spike rate
        :param odor_activation: (num) KC odor activation rate
        :param odor: (iterable) odor activation pattern
        :param init_KCMBONp: (num) initialization weights wKC_MBONp
        :param init_KCMBONn: (num) initialization weights wKC_MBONn
        :param init_KCDAN: (num) initialization weights wKCDAN
        :param KC: (np.array) KC activation pattern
        :param wKC_MBONp: (np.array) synaptic weight matrix
        :param wKC_MBONn: (np.array) synaptic weight matrix
        :param wKC_DAN: (np.array) synaptic weight matrix
        :param MBONp: (num) MBON activation
        :param MBONn: (num) MBON activation
        :param DAN: (num) DAN activation
        :param LR_KCMBONn: (num) learning rate
        :param DAN_activation: (num) DAN spike rate
        :param seed_param: (num) numpy seed for odor generation
        '''

    # compute KC odor activation

    if ref_odor == 'odor1':
        experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                   overlap=overlap, ref_odor=ref_odor, seed_param=seed_param)

    elif ref_odor == 'odor2':
        experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                   overlap=overlap, ref_odor=ref_odor, seed_param=seed_param)

    elif overlap_on == True:
        experiment_odors = odors.odor_input_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                    overlap=overlap, seed_param=seed_param)

    else:
        experiment_odors = odors.odor_input(KC_baseline, num_KC, odor_activation, seed_param)


    KC = experiment_odors[odor]
    # compute MBON output
    MBONp = np.dot(KC, wKC_MBONp)
    MBONn = np.dot(KC, wKC_MBONn)

    # compute DAN activation
    DAN = DAN_activation + np.dot(KC, wKC_DAN)
    if DAN < 0:
        DAN = 0

    # plasticity wKC_MBONn (coincidence of odor and reinforcement triggers plasticity)
    active_KC = np.where(KC > KC_baseline)[0]

    for neuron in active_KC:
        if wKC_MBONn[neuron] > (0 + LR_KCMBONn * DAN):  # ensure that weight never become negative
            wKC_MBONn[neuron] -= LR_KCMBONn * DAN

        if wKC_MBONn[neuron] <= (0 + LR_KCMBONn * DAN):
            wKC_MBONn[neuron] = 0

    # plasticity wKC_DAN
    for neuron in active_KC:
        wKC_DAN[neuron] += LR_KCDAN * DAN


    return KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wKC_DAN


def single_trial_model3(num_KC, KC_baseline, odor_activation, odor, init_KCMBONp, init_KCMBONn, init_MBONpDAN,init_MBONnDAN, KC, wKC_MBONp, wKC_MBONn, wMBONp_DAN,wMBONn_DAN, DAN, LR_KCMBONn, DAN_activation,seed_param,ref_odor=None,overlap=0,overlap_on=False):
    '''
        simulates a single trial

        :param num_KC: (int) number of KC neurons
        :param KC_baseline: (num) KC baseline spike rate
        :param odor_activation: (num) KC odor activation rate
        :param odor: (iterable) odor activation pattern
        :param init_KCMBONp: (num) initialization weights wKC_MBONp
        :param init_KCMBONn: (num) initialization weights wKC_MBONn
        :param init_MBONpDAN: (num) initialization weights wMBONpDAN
        :param KC: (np.array) KC activation pattern
        :param wKC_MBONp: (np.array) synaptic weight matrix
        :param wKC_MBONn: (np.array) synaptic weight matrix
        :param wMBONp_DAN: (np.array) synaptic weight matrix
        :param MBONp: (num) MBON activation
        :param MBONn: (num) MBON activation
        :param DAN: (num) DAN activation
        :param LR_KCMBONn: (num) learning rate
        :param DAN_activation: (num) DAN spike rate
        :param seed_param: (num) numpy seed for odor generation

        :return:
        KC (np.array) KC spike rates
        DAN (np.array) DAN spike rates
        MBONp (np.array) MBON spike rates
        MBONn (np.array) MBON spike rates
        weights (dict) all synaptic weights

        '''

    # compute KC odor activation

    if ref_odor == 'odor1':
        experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                   overlap=overlap, ref_odor=ref_odor, seed_param=seed_param)

    elif ref_odor == 'odor2':
        experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                   overlap=overlap, ref_odor=ref_odor, seed_param=seed_param)

    elif overlap_on == True:
        experiment_odors = odors.odor_input_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                    overlap=overlap, seed_param=seed_param)

    else:
        experiment_odors = odors.odor_input(KC_baseline, num_KC, odor_activation, seed_param)

    KC = experiment_odors[odor]
    # compute MBON output
    MBONp = np.dot(KC, wKC_MBONp)
    MBONn = np.dot(KC, wKC_MBONn)

    # compute DAN activation
    DAN = DAN_activation + np.dot(MBONp, wMBONp_DAN) - np.dot(MBONn, wMBONn_DAN)
    if DAN < 0:
        DAN = 0

    # plasticity wKC_MBONn (coincidence of odor and reinforcement triggers plasticity)
    active_KC = np.where(KC > KC_baseline)[0]

    for neuron in active_KC:
        if wKC_MBONn[neuron] > (0 + LR_KCMBONn * DAN):  # ensure that weight never become negative
            wKC_MBONn[neuron] -= LR_KCMBONn * DAN

        if wKC_MBONn[neuron] <= (0 + LR_KCMBONn * DAN):
            wKC_MBONn[neuron] = 0


    return KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wMBONp_DAN, wMBONn_DAN


def single_trial_model4(num_KC, KC_baseline, odor_activation, odor, init_KCMBONp, init_KCMBONn,init_MBONnDAN, KC, wKC_MBONp, wKC_MBONn, wMBONn_DAN, DAN, LR_KCMBONn, DAN_activation, DAN_baseline,seed_param,ref_odor=None,overlap=0,overlap_on=False):
    '''
        simulates a single trial

        :param num_KC: (int) number of KC neurons
        :param KC_baseline: (num) KC baseline spike rate
        :param odor_activation: (num) KC odor activation rate
        :param odor: (iterable) odor activation pattern
        :param init_KCMBONp: (num) initialization weights wKC_MBONp
        :param init_KCMBONn: (num) initialization weights wKC_MBONn
        :param init_MBONpDAN: (num) initialization weights wMBONpDAN
        :param KC: (np.array) KC activation pattern
        :param wKC_MBONp: (np.array) synaptic weight matrix
        :param wKC_MBONn: (np.array) synaptic weight matrix
        :param wMBONp_DAN: (np.array) synaptic weight matrix
        :param MBONp: (num) MBON activation
        :param MBONn: (num) MBON activation
        :param DAN: (num) DAN activation
        :param LR_KCMBONn: (num) learning rate
        :param DAN_activation: (num) DAN spike rate
        :param seed_param: (num) numpy seed for odor generation

        :return:
        KC (np.array) KC spike rates
        DAN (np.array) DAN spike rates
        MBONp (np.array) MBON spike rates
        MBONn (np.array) MBON spike rates
        weights (dict) all synaptic weights

        '''

    # compute KC odor activation

    if ref_odor == 'odor1':
        experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                   overlap=overlap, ref_odor=ref_odor, seed_param=seed_param)

    elif ref_odor == 'odor2':
        experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                   overlap=overlap, ref_odor=ref_odor, seed_param=seed_param)

    elif overlap_on == True:
        experiment_odors = odors.odor_input_overlap(KC_baseline, num_KC, odor_activation, num_KC_active=200,
                                                    overlap=overlap, seed_param=seed_param)

    else:
        experiment_odors = odors.odor_input(KC_baseline, num_KC, odor_activation, seed_param)

    KC = experiment_odors[odor]
    # compute MBON output
    MBONp = np.dot(KC, wKC_MBONp)
    MBONn = np.dot(KC, wKC_MBONn)

    # compute DAN activation
    DAN = DAN_activation + DAN_baseline - np.dot(MBONn, wMBONn_DAN)
    if DAN < 0:
        DAN = 0

    # plasticity wKC_MBONn (coincidence of odor and reinforcement triggers plasticity)
    active_KC = np.where(KC > KC_baseline)[0]

    for neuron in active_KC:
        if wKC_MBONn[neuron] > (0 + LR_KCMBONn * DAN):  # ensure that weight never become negative
            wKC_MBONn[neuron] -= LR_KCMBONn * DAN

        if wKC_MBONn[neuron] <= (0 + LR_KCMBONn * DAN):
            wKC_MBONn[neuron] = 0


    return KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wMBONn_DAN




def single_trial_model5(num_KC, KC_baseline, odor_activation, odor, init_KCMBONp, init_KCMBONn, init_MBONpDAN, KC, wKC_MBONp, wKC_MBONn, wMBONp_DAN, DAN, LR_KCMBONn,LR_MBONpDAN, DAN_activation,seed_param,ref_odor=None,overlap=0,overlap_on=False):
    '''
        simulates a single trial

        :param num_KC: (int) number of KC neurons
        :param KC_baseline: (num) KC baseline spike rate
        :param odor_activation: (num) KC odor activation rate
        :param odor: (iterable) odor activation pattern
        :param init_KCMBONp: (num) initialization weights wKC_MBONp
        :param init_KCMBONn: (num) initialization weights wKC_MBONn
        :param init_MBONpDAN: (num) initialization weights wMBONpDAN
        :param KC: (np.array) KC activation pattern
        :param wKC_MBONp: (np.array) synaptic weight matrix
        :param wKC_MBONn: (np.array) synaptic weight matrix
        :param wMBONp_DAN: (np.array) synaptic weight matrix
        :param MBONp: (num) MBON activation
        :param MBONn: (num) MBON activation
        :param DAN: (num) DAN activation
        :param LR_KCMBONn: (num) learning rate
        :param DAN_activation: (num) DAN spike rate
        :param seed_param: (num) numpy seed for odor generation

        :return:
        KC (np.array) KC spike rates
        DAN (np.array) DAN spike rates
        MBONp (np.array) MBON spike rates
        MBONn (np.array) MBON spike rates
        weights (dict) all synaptic weights

        '''



    # compute KC odor activation

    if ref_odor == 'odor1':
        experiment_odors = odors.odor_test_overlap(KC_baseline,num_KC,odor_activation,num_KC_active=200,overlap=overlap,ref_odor=ref_odor,seed_param=seed_param)

    elif ref_odor == 'odor2':
        experiment_odors = odors.odor_test_overlap(KC_baseline,num_KC,odor_activation,num_KC_active=200,overlap=overlap,ref_odor=ref_odor,seed_param=seed_param)

    elif overlap_on == True:
        experiment_odors = odors.odor_input_overlap(KC_baseline,num_KC,odor_activation,num_KC_active=200,overlap=overlap,seed_param=seed_param)

    else:
        experiment_odors = odors.odor_input(KC_baseline, num_KC, odor_activation, seed_param)



    KC = experiment_odors[odor]
    # compute MBON output
    MBONp = np.dot(KC, wKC_MBONp)
    MBONn = np.dot(KC, wKC_MBONn)

    # compute DAN activation
    DAN = DAN_activation + np.dot(MBONp, wMBONp_DAN)
    if DAN < 0:
        DAN = 0

    # plasticity wKC_MBONn (coincidence of odor and reinforcement triggers plasticity)
    active_KC = np.where(KC > KC_baseline)[0]


    for neuron in active_KC:
        if wKC_MBONn[neuron] > (0 + LR_KCMBONn * DAN): # ensure that weight never become negative
            wKC_MBONn[neuron] -= LR_KCMBONn * DAN

        if wKC_MBONn[neuron] <= (0 + LR_KCMBONn * DAN):
            wKC_MBONn[neuron] = 0


    # plasticity wMBONp_DAN
    if MBONp > 0:

        wMBONp_DAN += LR_MBONpDAN * DAN

    return KC, DAN, MBONp, MBONn, wKC_MBONp, wKC_MBONn, wMBONp_DAN


