import numpy as np
import sys


def create_model1(num_KC,init_KCMBONp,init_KCMBONn,init_KCKC,init_KCDAN):
    '''

      Creates KC matrix, MBONp, MBONn, DAN and initializes wKC_MBONp and wKC_MBONn

      :param num_KC: size of KC population (int)
      :param init_KCMBONp: initial weight for all connection (num)
      :param init_KCMBONn: initial weight for all connection (num)
      :param init_KCKC: initial weight (num)
      :param init_KCDAN: initial weight (num)

      :return: KC, wKC_MBONp, wKC_MBONn,wKC_DAN, wKCKC, MBONp, MBONn, DAN

      '''

    # create network structure
    KC = np.zeros((num_KC))
    MBONp = 0
    MBONn = 0
    DAN = 0

    # initialize weights
    wKC_MBONp = np.ones((num_KC)) * init_KCMBONp
    wKC_MBONn = np.ones((num_KC)) * init_KCMBONn
    wKC_DAN = np.ones((num_KC)) * init_KCDAN
    wKCKC = np.ones((num_KC,num_KC)) * init_KCKC

    return KC, wKC_MBONp, wKC_MBONn,wKC_DAN, wKCKC, MBONp, MBONn, DAN


def create_model2(num_KC,init_KCMBONp,init_KCMBONn,init_KCDAN):
    '''

      Creates KC matrix, MBONp, MBONn, DAN and initializes weight matrices

      :param num_KC: size of KC population (int)
      :param init_KCMBONp: initial weight for all connection (num)
      :param init_KCMBONn: initial weight for all connection (num)
      :param init_KCDAN: initial weight (num)

      :return: KC, wKC_MBONp, wKC_MBONn, wKC_DAN, MBONp, MBONn, DAN

      '''

    # create network structure
    KC = np.zeros((num_KC))
    MBONp = 0
    MBONn = 0
    DAN = 0

    # initialize weights
    wKC_MBONp = np.ones((num_KC)) * init_KCMBONp
    wKC_MBONn = np.ones((num_KC)) * init_KCMBONn
    wKC_DAN = np.ones((num_KC)) * init_KCDAN

    return KC, wKC_MBONp, wKC_MBONn, wKC_DAN, MBONp, MBONn, DAN


def create_model3(num_KC,init_KCMBONp,init_KCMBONn,init_MBONpDAN,init_MBONnDAN):
    '''

      Creates KC matrix, MBONp, MBONn, DAN and initializes wKC_MBONp and wKC_MBONn

      :param num_KC: size of KC population (int)
      :param init_KCMBONp: initial weight for all connection (num)
      :param init_KCMBONn: initial weight for all connection (num)
      :param init_MBONpDAN: initial weight (num)
      :param init_MBONnDAN: initial weight (num)

      :return: KC, wKC_MBONp, wKC_MBONn, wMBONp_DAN,wMBONn_DAN, MBONp, MBONn, DAN

      '''

    # create network structure
    KC = np.zeros((num_KC))
    MBONp = 0
    MBONn = 0
    DAN = 0

    # initialize weights
    wKC_MBONp = np.ones((num_KC)) * init_KCMBONp
    wKC_MBONn = np.ones((num_KC)) * init_KCMBONn
    wMBONp_DAN = init_MBONpDAN
    wMBONn_DAN = init_MBONnDAN

    return KC, wKC_MBONp, wKC_MBONn, wMBONp_DAN,wMBONn_DAN, MBONp, MBONn, DAN


def create_model4_old(num_KC,init_KCMBONp,init_KCMBONn,init_MBONDAN,init_KCMBON,init_MBONnMBON,init_MBONnDAN):
    '''

      Creates KC matrix, MBONp, MBONn, DAN and initializes wKC_MBONp and wKC_MBONn

      :param num_KC: size of KC population (int)
      :param init_KCMBONp: initial weight for all connection (num)
      :param init_KCMBONn: initial weight for all connection (num)
      :param init_MBONpDAN: initial weight (num)
      :param init_MBONnDAN: initial weight (num)

      :return: KC, wKC_MBONp, wKC_MBONn, wKC_MBON, wMBON_DAN,wMBONn_MBON, wMBONn_DAN,MBONp, MBONn,MBON, DAN

      '''

    # create network structure
    KC = np.zeros((num_KC))
    MBONp = 0
    MBONn = 0
    MBON = 0
    DAN = 0

    # initialize weights
    wKC_MBONp = np.ones((num_KC)) * init_KCMBONp
    wKC_MBONn = np.ones((num_KC)) * init_KCMBONn
    wKC_MBON = np.ones((num_KC)) * init_KCMBON
    wMBON_DAN = init_MBONDAN
    wMBONn_MBON = init_MBONnMBON
    wMBONn_DAN = init_MBONnDAN

    return KC, wKC_MBONp, wKC_MBONn, wKC_MBON, wMBON_DAN,wMBONn_MBON, wMBONn_DAN,MBONp, MBONn,MBON, DAN


def create_model4(num_KC,init_KCMBONp,init_KCMBONn,init_MBONnDAN,DAN_baseline):
    '''

      Creates KC matrix, MBONp, MBONn, DAN and initializes wKC_MBONp and wKC_MBONn

      :param num_KC: size of KC population (int)
      :param init_KCMBONp: initial weight for all connection (num)
      :param init_KCMBONn: initial weight for all connection (num)
      :param init_MBONpDAN: initial weight (num)
      :param init_MBONnDAN: initial weight (num)

      :return: KC, wKC_MBONp, wKC_MBONn, wMBONp_DAN,wMBONn_DAN, MBONp, MBONn, DAN

      '''

    # create network structure
    KC = np.zeros((num_KC))
    MBONp = 0
    MBONn = 0
    DAN = DAN_baseline

    # initialize weights
    wKC_MBONp = np.ones((num_KC)) * init_KCMBONp
    wKC_MBONn = np.ones((num_KC)) * init_KCMBONn
    wMBONn_DAN = init_MBONnDAN

    return KC, wKC_MBONp, wKC_MBONn, wMBONn_DAN, MBONp, MBONn, DAN, DAN_baseline


def create_model5(num_KC,init_KCMBONp,init_KCMBONn,init_MBONpDAN):
    '''

      Creates KC matrix, MBONp, MBONn, DAN and initializes weight matrices

      :param num_KC: size of KC population (int)
      :param init_KCMBONp: initial weight for all connection (num)
      :param init_KCMBONn: initial weight for all connection (num)
      :param init_MBONpDAN: initial weight (num)

      :return: KC, wKC_MBONp, wKC_MBONn, wMBONp_DAN, MBONp, MBONn, DAN

      '''

    # create network structure
    KC = np.zeros((num_KC))
    MBONp = 0
    MBONn = 0
    DAN = 0

    # initialize weights
    wKC_MBONp = np.ones((num_KC)) * init_KCMBONp
    wKC_MBONn = np.ones((num_KC)) * init_KCMBONn

    wMBONp_DAN = init_MBONpDAN

    return KC, wKC_MBONp, wKC_MBONn, wMBONp_DAN, MBONp, MBONn, DAN


