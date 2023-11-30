import numpy as np



def odor_input(KC_baseline,num_KC,odor_activation,seed_param):
    '''
    Creates KC odor activation patterns

    :param KC_baseline: baseline activation rate (num)
    :param num_KC: size of the KC population (int)
    :param odor_activation: odor activation rate (num)
    :param seed_param: (num) np.seed

    :return: odors (baseline + odor) (dict)
    '''

    np.random.seed(seed_param)
    KC_index = np.arange(0, num_KC)
    odor_sample = np.random.choice(KC_index, 1200, replace=False)

    index_odor1 = odor_sample[0:200]
    index_odor2 = odor_sample[200:400]
    index_odor3 = odor_sample[400:600]
    index_odor4 = odor_sample[600:800]
    index_odor5 = odor_sample[800:1000]
    index_odor6 = odor_sample[1000:1200]

    index_odor1_2_1 = np.random.choice(index_odor1, 100, replace=False)
    index_odor1_2_2 = np.random.choice(index_odor2, 100, replace=False)
    index_odor2_4_1 = np.random.choice(index_odor2, 100, replace=False)
    index_odor2_4_2 = np.random.choice(index_odor4, 100, replace=False)
    index_odor4_5_1 = np.random.choice(index_odor4, 100, replace=False)
    index_odor4_5_2 = np.random.choice(index_odor5, 100, replace=False)
    index_odor5_6_1 = np.random.choice(index_odor5, 100, replace=False)
    index_odor5_6_2 = np.random.choice(index_odor6, 100, replace=False)

    odor1 = np.zeros((num_KC)) + KC_baseline
    odor1[index_odor1] = odor_activation

    odor2 = np.zeros((num_KC)) + KC_baseline
    odor2[index_odor2] = odor_activation

    odor3 = np.zeros((num_KC)) + KC_baseline
    odor3[index_odor3] = odor_activation

    odor4 = np.zeros((num_KC)) + KC_baseline
    odor4[index_odor4] = odor_activation

    odor5 = np.zeros((num_KC)) + KC_baseline
    odor5[index_odor5] = odor_activation

    odor6 = np.zeros((num_KC)) + KC_baseline
    odor6[index_odor6] = odor_activation


    odor1_2 = np.zeros((num_KC)) + KC_baseline
    odor1_2[index_odor1_2_1] = odor_activation
    odor1_2[index_odor1_2_2] = odor_activation

    odor2_4 = np.zeros((num_KC)) + KC_baseline
    odor2_4[index_odor2_4_1] = odor_activation
    odor2_4[index_odor2_4_2] = odor_activation

    odor4_5 = np.zeros((num_KC)) + KC_baseline
    odor4_5[index_odor4_5_1] = odor_activation
    odor4_5[index_odor4_5_2] = odor_activation

    odor5_6 = np.zeros((num_KC)) + KC_baseline
    odor5_6[index_odor5_6_1] = odor_activation
    odor5_6[index_odor5_6_2] = odor_activation

    odors = {'odor1': odor1, 'odor2': odor2,'odor3': odor3,'odor4': odor4,'odor5': odor5,'odor6': odor6,'odor1_2': odor1_2,'odor2_4': odor2_4,'odor4_5': odor4_5,'odor5_6': odor5_6}

    return odors








def odor_input_overlap(KC_baseline,num_KC,odor_activation,num_KC_active,overlap,seed_param):

    '''
    Creates KC odor activation patterns for overlapping odors

    :param KC_baseline: baseline activation rate (num)
    :param num_KC: size of the KC population (int)
    :param odor_activation: odor activation rate (num)
    :param num_KC_active: number active KC per odor (num)
    :param overlap: percentage KC overlapping (num)
    :param seed_param: (num) np.seed

    :return: odors (baseline + odor) (dict)
    '''

    np.random.seed(seed_param)
    KC_index = np.arange(0, num_KC) # all KC
    num_KC_overlap = int(overlap/100 * num_KC_active)

    # create arrays to store odor patterns
    odor1 = np.zeros((num_KC)) + KC_baseline
    odor2 = np.zeros((num_KC)) + KC_baseline
    odor1_2 = np.zeros((num_KC)) + KC_baseline

    index_odor1 = np.random.choice(KC_index, num_KC_active,replace=False) #odor1 KCs

    not_odor1 = np.ones(num_KC, dtype=bool)
    not_odor1[index_odor1] = False
    index_odor2 = np.random.choice(index_odor1,num_KC_overlap,replace=False) #odor2 KCs overlapping with odor1
    index_odor2 = np.append(index_odor2,np.random.choice(KC_index[not_odor1], num_KC_active - num_KC_overlap,replace=False))  # add odor2 KCs not overlapping with odor 1

    # create compound odor 1 & 2 --> retain overlap in compound odor representation
    index_sharedKC = np.intersect1d(index_odor1, index_odor2)
    KC_exclusive_odor1 = np.zeros(num_KC, dtype=bool)
    KC_exclusive_odor1[index_odor1] = True
    KC_exclusive_odor1[index_sharedKC] = False

    KC_exclusive_odor2 = np.zeros(num_KC, dtype=bool)
    KC_exclusive_odor2[index_odor2] = True
    KC_exclusive_odor2[index_sharedKC] = False

    index_odor1_2 = np.random.choice(index_sharedKC, int(((overlap / 100) * num_KC_active)),replace=False)  # pick overlap/100 * num_KC_active KC from the pool of KC shared between odor1 and odor2
    remainder = num_KC_active - len(index_odor1_2)
    index_odor1_2 = np.append(index_odor1_2,np.random.choice(KC_index[KC_exclusive_odor1],int(remainder/2),replace=False))
    index_odor1_2 = np.append(index_odor1_2,np.random.choice(KC_index[KC_exclusive_odor2],int(remainder/2),replace=False))
    # create odor activation patterns
    odor1[index_odor1] = odor_activation
    odor2[index_odor2] = odor_activation
    odor1_2[index_odor1_2] = odor_activation

    odors = {'odor1': odor1, 'odor2': odor2, 'odor1_2': odor1_2}


    return odors





def odor_test_overlap(KC_baseline,num_KC,odor_activation,num_KC_active,overlap,ref_odor,seed_param):

    '''
    Creates KC odor activation patterns for test odor overlapping with training odor

    :param KC_baseline: baseline activation rate (num)
    :param num_KC: size of the KC population (int)
    :param odor_activation: odor activation rate (num)
    :param num_KC_active: number active KC per odor (num)
    :param overlap: percentage KC overlapping (num)
    :param reference_odor: odor that should be overlapping with odor3, either odor1 or odor2 (str)
    :param seed_param: (num) np.seed

    :return: odors (baseline + odor) (dict)
    '''

    np.random.seed(seed_param)
    KC_index = np.arange(0, num_KC)
    odor_sample = np.random.choice(KC_index, 600, replace=False)

    index_odor1 = odor_sample[0:200]
    index_odor2 = odor_sample[200:400]
    index_odor1_2_1 = np.random.choice(index_odor1, 100, replace=False)
    index_odor1_2_2 = np.random.choice(index_odor2, 100, replace=False)

    odor1 = np.zeros((num_KC)) + KC_baseline
    odor1[index_odor1] = odor_activation

    odor2 = np.zeros((num_KC)) + KC_baseline
    odor2[index_odor2] = odor_activation

    odor1_2 = np.zeros((num_KC)) + KC_baseline
    odor1_2[index_odor1_2_1] = odor_activation
    odor1_2[index_odor1_2_2] = odor_activation


    # # generate odor 3
    num_KC_overlap = int(overlap / 100 * num_KC_active)
    not_odor1or2 = np.ones(num_KC, dtype=bool)
    not_odor1or2[np.r_[index_odor1, index_odor2]] = False

    if ref_odor == 'odor1':
        index_odor3 = np.random.choice(index_odor1, num_KC_overlap, replace=False)  # odor3 KCs overlapping with reference_odor
        index_odor3 = np.append(index_odor3, np.random.choice(KC_index[not_odor1or2], num_KC_active - num_KC_overlap,replace=False))  # add odor3 KCs not overlapping with reference_odor

    if ref_odor == 'odor2':
        index_odor3 = np.random.choice(index_odor2, num_KC_overlap, replace=False)  # odor3 KCs overlapping with reference_odor
        index_odor3 = np.append(index_odor3, np.random.choice(KC_index[not_odor1or2], num_KC_active - num_KC_overlap,replace=False))  # add odor3 KCs not overlapping with reference_odor


    odor3 = np.zeros((num_KC)) + KC_baseline
    odor3[index_odor3] = odor_activation

    odors = {'odor1': odor1, 'odor2': odor2, 'odor1_2': odor1_2, 'odor3': odor3}


    return odors


# odors = odor_test_overlap(0,2000,1,200,0,'odor1',999)
# print(len(np.intersect1d(np.where(odors['odor1']==1),np.where(odors['odor1']==1))))
# print(len(np.intersect1d(np.where(odors['odor1']==1),np.where(odors['odor2']==1))))
# print(len(np.intersect1d(np.where(odors['odor1']==1),np.where(odors['odor1_2']==1))))
# print(len(np.intersect1d(np.where(odors['odor2']==1),np.where(odors['odor3']==1))))
