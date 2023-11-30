import numpy as np
import odors


def test_model1(num_KC, KC_baseline, odor_activation, test_odor, wKC_MBONp, wKC_MBONn,seed_param):

    '''

    test the models odor preference

    :param num_KC: number of KCs (num)
    :param KC_baseline: KC baseline spike rate (num)
    :param odor_activation: KC odor activation level (num)
    :param test_odor: odor (str)
    :param wKC_MBONp: weight KC-MBONp synapse (num)
    :param wKC_MBONn: weight KC-MBONn synapse (num)
    :param seed_param: seed

    :return: odor preference (num)
    '''

    # compute KC odor activation
    experiment_odors = odors.odor_input(KC_baseline, num_KC, odor_activation, seed_param)

    KC = experiment_odors[test_odor]
    # compute MBON output
    MBONp = np.dot(KC, wKC_MBONp)
    MBONn = np.dot(KC, wKC_MBONn)

    pref = (MBONp-MBONn)/(MBONp+MBONn)

    return pref

def test_model_overlap(num_KC, KC_baseline, odor_activation, test_odor, wKC_MBONp, wKC_MBONn,overlap,seed_param):
    '''

      test the models odor preference

      :param num_KC: number of KCs (num)
      :param KC_baseline: KC baseline spike rate (num)
      :param odor_activation: KC odor activation level (num)
      :param test_odor: odor (str)
      :param wKC_MBONp: weight KC-MBONp synapse (num)
      :param wKC_MBONn: weight KC-MBONn synapse (num)
      :param seed_param: seed

      :return: odor preference (num)
      '''

    # compute KC odor activation
    experiment_odors = odors.odor_input_overlap(KC_baseline, num_KC, odor_activation,200,overlap=overlap,seed_param=seed_param)

    KC = experiment_odors[test_odor]
    # compute MBON output
    MBONp = np.dot(KC, wKC_MBONp)
    MBONn = np.dot(KC, wKC_MBONn)

    pref = (MBONp-MBONn)/(MBONp+MBONn)

    return pref


def test_model_overlap_test(num_KC, KC_baseline, odor_activation, test_odor, wKC_MBONp, wKC_MBONn,overlap,ref_odor,seed_param):
    '''

      test the models odor preference

      :param num_KC: number of KCs (num)
      :param KC_baseline: KC baseline spike rate (num)
      :param odor_activation: KC odor activation level (num)
      :param test_odor: odor (str)
      :param wKC_MBONp: weight KC-MBONp synapse (num)
      :param wKC_MBONn: weight KC-MBONn synapse (num)
      :param seed_param: seed

      :return: odor preference (num)
      '''

    # compute KC odor activation
    experiment_odors = odors.odor_test_overlap(KC_baseline, num_KC, odor_activation,200,overlap=overlap,ref_odor=ref_odor,seed_param=seed_param)

    KC = experiment_odors[test_odor]
    # compute MBON output
    MBONp = np.dot(KC, wKC_MBONp)
    MBONn = np.dot(KC, wKC_MBONn)

    pref = (MBONp-MBONn)/(MBONp+MBONn)

    return pref