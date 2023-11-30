import numpy as np
import run_model1
import run_model2
import run_model3
import run_model4
import run_model5
import sys

# # run experiments with all models with variable KC-MBON learning rate

sample_size = 100

minLR = 0.001
maxLR = 0.004


## model 5
post_foc_odor1 = np.zeros(sample_size)
post_foc_odor2 = np.zeros(sample_size)
post_foc_odor3 = np.zeros(sample_size)
post_soc_odor1 = np.zeros(sample_size)
post_soc_odor2 = np.zeros(sample_size)
post_soc_odor3 = np.zeros(sample_size)

LR_model1 = np.random.uniform(minLR, maxLR, size=sample_size)

for i in (np.arange(sample_size)):

    DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, KC_KC_weights, KC_DAN_weights, test_results = run_model1.run(seed_param=np.random.randint(1,999,1),trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0, odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
             init_KCMBONn=0.083, init_KCKC=0, init_KCDAN=0.001000, LR_KCMBONn=LR_model1[i],LR_KCKC=0.000162, DAN_activation=7.272727,variable_weight=False)



    post_foc_odor1[i] = test_results['post FOC odor1']
    post_foc_odor2[i] = test_results['post FOC odor2']
    post_foc_odor3[i] = test_results['post FOC odor3']
    post_soc_odor1[i] = test_results['post SOC odor1']
    post_soc_odor2[i] = test_results['post SOC odor2']
    post_soc_odor3[i] = test_results['post SOC odor3']

results1 = {}
results1.update({'post FOC odor1': post_foc_odor1})
results1.update({'post FOC odor2': post_foc_odor2})
results1.update({'post FOC odor3': post_foc_odor3})
results1.update({'post SOC odor1': post_soc_odor1})
results1.update({'post SOC odor2': post_soc_odor2})
results1.update({'post SOC odor3': post_soc_odor3})


np.save('/Users/anna/Documents/Python/Data/results1', results1)



## model 2
post_foc_odor1 = np.zeros(sample_size)
post_foc_odor2 = np.zeros(sample_size)
post_foc_odor3 = np.zeros(sample_size)
post_soc_odor1 = np.zeros(sample_size)
post_soc_odor2 = np.zeros(sample_size)
post_soc_odor3 = np.zeros(sample_size)

LR_model2 = np.random.uniform(minLR, maxLR, size=sample_size)

for i in (np.arange(sample_size)):
    DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, KC_DAN_weights, test_results = run_model2.run(seed_param=np.random.randint(1,999,1),trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0, odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
                 init_KCMBONn=0.083, initKCDAN=0.000505 , LR_KCMBONn=LR_model2[i], LR_KCDAN=0.000677,DAN_activation=5.727273,variable_weight=False)

    post_foc_odor1[i] = test_results['post FOC odor1']
    post_foc_odor2[i] = test_results['post FOC odor2']
    post_foc_odor3[i] = test_results['post FOC odor3']
    post_soc_odor1[i] = test_results['post SOC odor1']
    post_soc_odor2[i] = test_results['post SOC odor2']
    post_soc_odor3[i] = test_results['post SOC odor3']

results2 = {}
results2.update({'post FOC odor1': post_foc_odor1})
results2.update({'post FOC odor2': post_foc_odor2})
results2.update({'post FOC odor3': post_foc_odor3})
results2.update({'post SOC odor1': post_soc_odor1})
results2.update({'post SOC odor2': post_soc_odor2})
results2.update({'post SOC odor3': post_soc_odor3})


np.save('/Users/anna/Documents/Python/Data/results2', results2)


## model 3
post_foc_odor1 = np.zeros(sample_size)
post_foc_odor2 = np.zeros(sample_size)
post_foc_odor3 = np.zeros(sample_size)
post_soc_odor1 = np.zeros(sample_size)
post_soc_odor2 = np.zeros(sample_size)
post_soc_odor3 = np.zeros(sample_size)

LR_model3 = np.random.uniform(minLR, maxLR, size=sample_size)

for i in (np.arange(sample_size)):
    DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONp_DAN_weights, MBONn_DAN_weights, test_results = run_model3.run(seed_param=np.random.randint(1,999,1),trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0, odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
                 init_KCMBONn=0.083, initMBONpDAN=0.272727, initMBONnDAN=0.262626, LR_KCMBONn=LR_model3[i], DAN_activation=5,variable_weight=False)


    post_foc_odor1[i] = test_results['post FOC odor1']
    post_foc_odor2[i] = test_results['post FOC odor2']
    post_foc_odor3[i] = test_results['post FOC odor3']
    post_soc_odor1[i] = test_results['post SOC odor1']
    post_soc_odor2[i] = test_results['post SOC odor2']
    post_soc_odor3[i] = test_results['post SOC odor3']

results3 = {}
results3.update({'post FOC odor1': post_foc_odor1})
results3.update({'post FOC odor2': post_foc_odor2})
results3.update({'post FOC odor3': post_foc_odor3})
results3.update({'post SOC odor1': post_soc_odor1})
results3.update({'post SOC odor2': post_soc_odor2})
results3.update({'post SOC odor3': post_soc_odor3})


np.save('/Users/anna/Documents/Python/Data/results3', results3)


## model 4
post_foc_odor1 = np.zeros(sample_size)
post_foc_odor2 = np.zeros(sample_size)
post_foc_odor3 = np.zeros(sample_size)
post_soc_odor1 = np.zeros(sample_size)
post_soc_odor2 = np.zeros(sample_size)
post_soc_odor3 = np.zeros(sample_size)

LR_model4 = np.random.uniform(minLR, maxLR, size=sample_size)

for i in (np.arange(sample_size)):
    DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights,  MBONn_DAN_weights, test_results = run_model4.run(seed_param=np.random.randint(1,999,1),trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0, odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
                 init_KCMBONn=0.083, initMBONnDAN=0.131313, LR_KCMBONn=LR_model4[i], DANbaseline=11.212121, DAN_activation=3.727273,variable_weight=False)

    post_foc_odor1[i] = test_results['post FOC odor1']
    post_foc_odor2[i] = test_results['post FOC odor2']
    post_foc_odor3[i] = test_results['post FOC odor3']
    post_soc_odor1[i] = test_results['post SOC odor1']
    post_soc_odor2[i] = test_results['post SOC odor2']
    post_soc_odor3[i] = test_results['post SOC odor3']

results4 = {}
results4.update({'post FOC odor1': post_foc_odor1})
results4.update({'post FOC odor2': post_foc_odor2})
results4.update({'post FOC odor3': post_foc_odor3})
results4.update({'post SOC odor1': post_soc_odor1})
results4.update({'post SOC odor2': post_soc_odor2})
results4.update({'post SOC odor3': post_soc_odor3})


np.save('/Users/anna/Documents/Python/Data/results4', results4)


## model 5
post_foc_odor1 = np.zeros(sample_size)
post_foc_odor2 = np.zeros(sample_size)
post_foc_odor3 = np.zeros(sample_size)
post_soc_odor1 = np.zeros(sample_size)
post_soc_odor2 = np.zeros(sample_size)
post_soc_odor3 = np.zeros(sample_size)

LR_model5 = np.random.uniform(minLR, maxLR, size=sample_size)

for i in (np.arange(sample_size)):

    DAN_rate, MBONp_rate, MBONn_rate, KC_MBONp_weights, KC_MBONn_weights, MBONp_DAN_weights, test_results = run_model5.run(
        seed_param=np.random.randint(1, 999, 1), trials_FOC=3, trials_SOC=3, num_KC=2000, KC_baseline=0,
        odor_activation=3, odor_FOC='odor1', odor_SOC='odor1_2', init_KCMBONp=0.083,
        init_KCMBONn=0.083, initMBONpDAN=0.080808, LR_KCMBONn=LR_model5[i], LR_MBONpDAN=0.003000, DAN_activation=4.727273,
        variable_weight=False)


    post_foc_odor1[i] = test_results['post FOC odor1']
    post_foc_odor2[i] = test_results['post FOC odor2']
    post_foc_odor3[i] = test_results['post FOC odor3']
    post_soc_odor1[i] = test_results['post SOC odor1']
    post_soc_odor2[i] = test_results['post SOC odor2']
    post_soc_odor3[i] = test_results['post SOC odor3']

results5 = {}
results5.update({'post FOC odor1': post_foc_odor1})
results5.update({'post FOC odor2': post_foc_odor2})
results5.update({'post FOC odor3': post_foc_odor3})
results5.update({'post SOC odor1': post_soc_odor1})
results5.update({'post SOC odor2': post_soc_odor2})
results5.update({'post SOC odor3': post_soc_odor3})


np.save('/Users/anna/Documents/Python/Data/results5', results5)
