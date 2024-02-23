# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:04:23 2021

@author: maudb
"""
HPC = False

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from Functions2 import create_design, Incorrelation_repetition, groupdifference_repetition, check_input_parameters, Excorrelation_repetition
from scipy import stats as stat
from datetime import datetime

if HPC == False:
    import seaborn as sns
    import matplotlib.pyplot as plt

#This is to avoid warnings being printed to the terminal window
import warnings
warnings.filterwarnings('ignore')


def power_estimation_Incorrelation(npp = 30, ntrials = 480, nreversals = 12, cut_off = 0.7, high_performance = False,
                                 nreps = 100, reward_probability = 0.8, mean_LR1distribution = 0.5,mean_LR2distribution = 0.5, SD_LR1distribution = 0.1,
                                 SD_LR2distribution = 0.1, corr = 0.5, mean_inverseTempdistribution = 2.0, SD_inverseTempdistribution = 1.0):
    """ 

    Parameters
    ----------
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    cut_off : float
        Critical value that will be used to evaluate whether the repetition was successful.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.
    mean_LRdistribution: float
        Mean for the normal distribution to sample learning rates from.
    SD_LRdistribution: float
        Standard deviation for the normal distribution to sample learning rates from.
    mean_inverseTempdistribution: float
        Mean for the normal distribution to sample inverse temperatures from.
    SD_inverseTempdistribution: float
        Standard deviation for the normal distribution to sample inverse temperatures from.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the correlation value on each repetition.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which the parameter recovery was successful (correlation > significance_cutoff) divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the probability to obtain adequate parameter estimates.
    Parameter estimates are considered to be adequate if their correlation with the true parameters is minimum the cut_off.
    Power is calculated using a Monte Carlo simulation-based approach.
    """
    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)
    ##
    if HPC == True: n_cpu = cpu_count()
    elif high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1

    #divide process over multiple cores
    pool = Pool(processes = n_cpu)
    LR1_distribution = np.array([mean_LR1distribution, SD_LR1distribution])
    LR2_distribution = np.array([mean_LR2distribution, SD_LR2distribution])
    inverseTemp_distribution = np.array([mean_inverseTempdistribution, SD_inverseTempdistribution])
    out = pool.starmap(Incorrelation_repetition, [(inverseTemp_distribution, LR1_distribution, LR2_distribution, corr, npp, ntrials,
                                                 start_design, rep, nreps, n_cpu) for rep in range(nreps)])
    pool.close()
    pool.join()
    
    out_all = np.array(out)
    allreps_output_lr1 = pd.DataFrame(out_all[:, 0], columns = ['correlations'])
    allreps_output_lr2 = pd.DataFrame(out_all[:, 1], columns = ['correlations'])
    
    power_estimate_lr1 = np.mean((allreps_output_lr1['correlations'] >= cut_off)*1)
    power_estimate_lr2 = np.mean((allreps_output_lr2['correlations'] >= cut_off)*1)
    print(str("\nProbability to obtain a correlation(true_param_lr1, param_estim) >= {}".format(cut_off)
          + " with {} trials and {} participants: {}%".format(ntrials, npp, power_estimate_lr1*100)))
    print(str("\nProbability to obtain a correlation(true_param_lr2, param_estim) >= {}".format(cut_off)
          + " with {} trials and {} participants: {}%".format(ntrials, npp, power_estimate_lr2*100)))

    return allreps_output_lr1, allreps_output_lr2, power_estimate_lr1, power_estimate_lr2

def power_estimation_Excorrelation(npp = 100, ntrials = 480, nreversals = 12, typeIerror = 0.05, high_performance = False,
                                 nreps = 100, reward_probability = 0.8, mean_LR1distribution = 0.5, SD_LR1distribution = 0.1, mean_LR2distribution = 0.5,SD_LR2distribution = 0.1,
                                 mean_inverseTempdistribution = 2.0, SD_inverseTempdistribution = 1.0, True_correlation = [0.5, 0.5, 0.5]):
    """

    Parameters
    ----------
    npp : integer
        Number of participants in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    typeIerror : float
        Critical value for p-values. From this also the cut-off for the correlation statistic can be determined.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.
    mean_LRdistribution: float
        Mean for the normal distribution to sample learning rates from.
    SD_LRdistribution: float
        Standard deviation for the normal distribution to sample learning rates from.
    mean_inverseTempdistribution: float
        Mean for the normal distribution to sample inverse temperatures from.
    SD_inverseTempdistribution: float
        Standard deviation for the normal distribution to sample inverse temperatures from.
    True_correlation: numpy array (3,1)
        The hypothesized correlation between the learning rate and the external measure theta.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the correlation value on each repetition, the p-value and the p-value if estimates would be perfect.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which the parameter recovery was successful (correlation > significance_cutoff) divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the probability to obtain significant correlations with external measures.
    Parameter estimates are considered to be adequate if correctly reveal a significant correlation when a significant correlation.
    Power is calculated using a Monte Carlo simulation-based approach.
    """
    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)
    if HPC == True: n_cpu = cpu_count()
    elif high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1

    #Use beta_distribution to determine the p-value for the hypothesized correlation
    beta_distribution = stat.beta((npp/2)-1, (npp/2)-1, loc = -1, scale = 2)
    true_pValue1 = 1-beta_distribution.cdf(True_correlation[1])
    true_pValue2 = 1-beta_distribution.cdf(True_correlation[2])
    tau = -beta_distribution.ppf(typeIerror/2)
    
    #compute conventional power
    noncentral_beta1 = stat.beta((npp/2)-1, (npp/2)-1, loc = -1+True_correlation[1], scale = 2)
    conventional_power1 = 1-noncentral_beta1.cdf(tau)
    noncentral_beta2 = stat.beta((npp/2)-1, (npp/2)-1, loc = -1+True_correlation[2], scale = 2)
    conventional_power2 = 1-noncentral_beta2.cdf(tau)

    print(str("\nThe correlation cut-off value is: {}".format(np.round(tau,2))))
    print(str("\np-value for true correlation is :{}".format(np.round(true_pValue1,5))))
    print(str("\nProbability to obtain a significant correlation under conventional power implementation: {}%".format(np.round(conventional_power1*100,2))))
    print(str("\np-value for true correlation is :{}".format(np.round(true_pValue2,5))))
    print(str("\nProbability to obtain a significant correlation under conventional power implementation: {}%".format(np.round(conventional_power2*100,2))))
    
    #divide process over multiple cores
    pool = Pool(processes = n_cpu)
    LR1_distribution = np.array([mean_LR1distribution, SD_LR1distribution])
    LR2_distribution = np.array([mean_LR2distribution, SD_LR2distribution])
    inverseTemp_distribution = np.array([mean_inverseTempdistribution, SD_inverseTempdistribution])
    out = pool.starmap(Excorrelation_repetition, [(inverseTemp_distribution, LR1_distribution, LR2_distribution, True_correlation, npp, ntrials,
                                                 start_design, rep, nreps, n_cpu) for rep in range(nreps)])
    pool.close()
    pool.join()

    #allreps_output = pd.DataFrame(out, columns = ['Statistic','estimated_pValue', 'True_pValue'])
    out_all = np.array(out)
    allreps_output_lr1 = pd.DataFrame(out_all[:, [0,1]], columns = ['Statistic', 'estimated_pValue'])
    allreps_output_lr2 = pd.DataFrame(out_all[:, [4,5]], columns = ['Statistic', 'estimated_pValue'])
    
    #Compute power if estimates would be perfect.
    #power1_true = np.mean((allreps_output['True_pValue1'] <= typeIerror/2)*1)
    #print(str("\nProbability to obtain a significant correlation under conventional power implementation: {}%".format(np.round(power1_true*100,2))))
    #power2_true = np.mean((allreps_output['True_pValue2'] <= typeIerror/2)*1)
    #print(str("\nProbability to obtain a significant correlation under conventional power implementation: {}%".format(np.round(power2_true*100,2))))
    
    #Compute power for correlation with estimated parameter values.
    power_estimate_lr1 = np.mean((allreps_output_lr1['estimated_pValue'] <= typeIerror/2)*1)
    power_estimate_lr2 = np.mean((allreps_output_lr2['estimated_pValue'] <= typeIerror/2)*1)
    print(str("\nProbability to obtain a significant correlation between model parameter positive learning rate and an external measure that is {} correlated".format(True_correlation[1])
          + " with {} trials and {} participants: {}%".format(ntrials, npp, np.round(power_estimate_lr1*100,2))))
    print(str("\nProbability to obtain a significant correlation between model parameter negative learning rate and an external measure that is {} correlated".format(True_correlation[2])
          + " with {} trials and {} participants: {}%".format(ntrials, npp, np.round(power_estimate_lr2*100,2))))
    return allreps_output_lr1, power_estimate_lr1, allreps_output_lr2, power_estimate_lr2

def power_estimation_groupdifference(npp_per_group = 20, ntrials = 480, nreps = 100, typeIerror = 0.05,
                                     high_performance = False, nreversals = 12, reward_probability = 0.8,
                                     mean_LR1distributionG1 = 0.5, SD_LR1distributionG1 = 0.1,
                                     mean_LR1distributionG2 = 0.5, SD_LR1distributionG2 = 0.1,
                                     mean_LR2distributionG1 = 0.5, SD_LR2distributionG1 = 0.1,
                                     mean_LR2distributionG2 = 0.5, SD_LR2distributionG2 = 0.1,
                                     corr = 0.5, cohens_d1 = 0.5, cohens_d2 = 0.5,
                                     mean_inverseTempdistributionG1 = 2.0, SD_inverseTempdistributionG1 = 1.0,
                                     mean_inverseTempdistributionG2 = 2.0, SD_inverseTempdistributionG2 = 1.0):
    """
    Parameters
    ----------
    npp_per_group : integer
        Number of participants per group in the study.
    ntrials : integer
        Number of trials that will be used to do the parameter recovery analysis for each participant.
    nreps : integer
        Number of repetitions that will be used for the parameter estimation process.
    typeIerror : float
        Critical value for p-values. From this also the cut-off for the correlation statistic can be determined.
    high_performance : bool (True or False)
        Defines whether multiple cores on the computer will be used in order to estimate the power.
    nreversals : integer
        The number of rule-reversals that will occur in the experiment. Should be smaller than ntrials.
    reward_probability : float (element within [0, 1]), optional
        The probability that reward will be congruent with the current stimulus-response mapping rule. The default is 0.8.
    mean_LRdistributionG1: float
        Mean for the normal distribution to sample learning rates for group 1.
    SD_LRdistributioG1: float
        Standard deviation for the normal distribution to sample learning rates for group 1.
    mean_inverseTempdistributionG1: float
        Mean for the normal distribution to sample inverse temperatures for group 1.
    SD_inverseTempdistributionG1: float
        Standard deviation for the normal distribution to sample inverse temperatures for group 1.
    mean_LRdistributionG2: float
        Mean for the normal distribution to sample learning rates for group 2.
    SD_LRdistributioG2: float
        Standard deviation for the normal distribution to sample learning rates for group 2.
    mean_inverseTempdistributionG2: float
        Mean for the normal distribution to sample inverse temperatures for group 2.
    SD_inverseTempdistributionG2: float
        Standard deviation for the normal distribution to sample inverse temperatures for group 2.

    Returns
    -------
    allreps_output : TYPE
        Pandas dataframe containing the p-value on each repetition.
    power_estimate: float [0, 1]
        The power estimation: number of reps for which a significant group difference was found divided by the total number of reps.

    Description
    -----------
    Function that actually calculates the probability to obtain adequate parameter estimates.
    Parameter estimates are considered to be adequate if they correctly reveal the group difference when a true group difference of size 'cohens_d' exists.
    Power is calculated using a Monte Carlo simulation-based approach.
    """

    start_design = create_design(ntrials = ntrials, nreversals = nreversals, reward_probability = reward_probability)
    if high_performance == True: n_cpu = cpu_count() - 2
    else: n_cpu = 1
    if __name__ == '__main__':
        
        #Use t_distribution to determine the p-value for the hypothesized cohen's d
        true_pValue1 = 1-stat.t.cdf(cohens_d1*np.sqrt(npp_per_group), (npp_per_group-1)*2)
        true_pValue2 = 1-stat.t.cdf(cohens_d2*np.sqrt(npp_per_group), (npp_per_group-1)*2)
        tau = -stat.t.ppf(typeIerror/2, (npp_per_group-1)*2) # equal standard deviation
        
        
        #Compute conventional power
        conventional_power1 = 1-stat.nct.cdf(tau, (npp_per_group-1)*2, cohens_d1*np.sqrt(npp_per_group))
        conventional_power2 = 1-stat.nct.cdf(tau, (npp_per_group-1)*2, cohens_d2*np.sqrt(npp_per_group))
        
        print(str("\nThe t-distribution cut-off value is: {}".format(np.round(tau,2))))
        print(str("\np-value for given cohen's d1 is :{}".format(np.round(true_pValue1,5))))
        print(str("\np-value for given cohen's d2 is :{}".format(np.round(true_pValue2,5))))
        print("\nProbability to obtain a significant group difference under conventional power implementation: {}%".format(np.round(conventional_power1*100,2)))
        print("\nProbability to obtain a significant group difference under conventional power implementation: {}%".format(np.round(conventional_power2*100,2)))
        
        #divide process over multiple cores
        if mean_LR1distributionG1 > mean_LR1distributionG2:
            LR1_distributions = np.array([[mean_LR1distributionG1, SD_LR1distributionG1], [mean_LR1distributionG2, SD_LR1distributionG2]])
        else:
            LR1_distributions = np.array([[mean_LR1distributionG2, SD_LR1distributionG2], [mean_LR1distributionG1, SD_LR1distributionG1]])

        if mean_LR2distributionG1 > mean_LR2distributionG2:
            LR2_distributions = np.array([[mean_LR2distributionG1, SD_LR2distributionG1], [mean_LR1distributionG2, SD_LR1distributionG2]])
        else: 
            LR2_distributions = np.array([[mean_LR2distributionG2, SD_LR2distributionG2], [mean_LR2distributionG1, SD_LR2distributionG1]])

        inverseTemp_distributions = np.array([[mean_inverseTempdistributionG1, SD_inverseTempdistributionG1],
                                              [mean_inverseTempdistributionG2, SD_inverseTempdistributionG2]])
        pool = Pool(processes = n_cpu)
        out = pool.starmap(groupdifference_repetition, [(inverseTemp_distributions, LR1_distributions,LR2_distributions, corr, npp_per_group,
                                                     ntrials, start_design, rep, nreps, n_cpu, False) for rep in range(nreps)])
        # before calling pool.join(), should call pool.close() to indicate that there will be no new processing
        pool.close()
        pool.join()
        
        out_all = np.array(out)
        allreps_output_lr1 = pd.DataFrame(out_all[:, [0,1]], columns = ['Statistic', 'estimated_pValue'])
        allreps_output_lr2 = pd.DataFrame(out_all[:, [2,3]], columns = ['Statistic', 'estimated_pValue'])

        # check for which % of repetitions the group difference was significant
        # note that we're working with a one-sided t-test (if interested in two-sided need to divide the p-value obtained at each rep with 2)
        power_estimate1 = np.mean((allreps_output_lr1['estimated_pValue'] <= typeIerror/2))
        power_estimate2 = np.mean((allreps_output_lr2['estimated_pValue'] <= typeIerror/2))
        print(str("\nProbability to detect a significant group difference when the estimated effect size d = {}".format(np.round(cohens_d1,3))
              + " with {} trials and {} participants per group: {}%".format(ntrials,
                                                                         npp_per_group, np.round(power_estimate1*100,2))))
        print(str("\nProbability to detect a significant group difference when the estimated effect size d = {}".format(np.round(cohens_d2,3))
              + " with {} trials and {} participants per group: {}%".format(ntrials,
                                                                   npp_per_group, np.round(power_estimate2*100,2))))
        return allreps_output_lr1, power_estimate1, allreps_output_lr2, power_estimate2

#%%
import os, sys

if __name__ == '__main__':
    criterion = sys.argv[1:]
    assert len(criterion) == 1
    criterion = criterion[0]
    power_estimate = []
    
    InputFile_name = "InputFile2_{}s.csv".format(criterion)
    InputFile_path = os.path.join(os.getcwd(), InputFile_name)
    InputParameters = pd.read_csv(InputFile_path, delimiter = ',')
    if InputParameters.shape[1] == 1: InputParameters = pd.read_csv(InputFile_path, delimiter = ';')	# depending on how you save the csv-file, the delimiter should be "," or ";". - This if-statement ensures that the correct delimiter is used. 
    InputDictionary = InputParameters.to_dict()

    for row in range(InputParameters.shape[0]):
        #Calculate how long it takes to do a power estimation
        start_time = datetime.now()
        print("Power estimation started at {}.".format(start_time))
    
        #Extract all values that are the same regardless of the criterion used
        ntrials = InputDictionary['ntrials'][row]
        nreversals = InputDictionary['nreversals'][row]
        reward_probability = InputDictionary['reward_probability'][row]
        nreps = InputDictionary['nreps'][row]
        full_speed = InputDictionary['full_speed'][row]
        output_folder = InputDictionary['output_folder'][row]
        
        variables_fine = check_input_parameters(ntrials, nreversals, reward_probability, full_speed, criterion, output_folder)
        if variables_fine == 0: quit()
        
        #if not os.path.isdir(output_folder): 
        #    print('output_folder does not exist, please adapt the csv-file')
        #    quit()
    
        if criterion == "IC":
            npp = InputDictionary['npp'][row]
            meanLR1, sdLR1 = InputDictionary['meanLR1'][row], InputDictionary['sdLR1'][row]
            meanLR2, sdLR2 = InputDictionary['meanLR2'][row], InputDictionary['sdLR2'][row]
            corr = InputDictionary['corr'][row]
            meanInverseT, sdInverseT = InputDictionary['meanInverseTemperature'][row], InputDictionary['sdInverseTemperature'][row]
            tau = InputDictionary['tau'][row]
            s_pooled1 = sdLR1
            s_pooled2 = sdLR2
    
            output1, output2, power_estimate1, power_estimate2 = power_estimation_Incorrelation(npp = npp, ntrials = ntrials, nreps = nreps,
                                                                  cut_off = tau, corr = corr,
                                               high_performance = full_speed, nreversals = nreversals,
                                               reward_probability = reward_probability, mean_LR1distribution = meanLR1,
                                               SD_LR1distribution = sdLR1, mean_LR2distribution = meanLR2, SD_LR2distribution = sdLR2, mean_inverseTempdistribution = meanInverseT,
                                               SD_inverseTempdistribution = sdInverseT)
           
            output1.to_csv(os.path.join(output_folder, 'OutputIC_LR1{}SD{}T{}R{}N{}REP{}CORR{}REW.csv'.format(s_pooled1, ntrials,
                                                                                      nreversals,
                                                                                      npp, nreps, corr, reward_probability)))
            output2.to_csv(os.path.join(output_folder, 'OutputIC_LR2{}SD{}T{}R{}N{}REP{}CORR{}REW.csv'.format(s_pooled2, ntrials,
                                                                                        nreversals,
                                                                                        npp, nreps, corr, reward_probability)))   
            if HPC == False:
                fig1, axes1 = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output1["correlations"], label = "Correlations", ax = axes1, cut = 0)
                fig1.suptitle("Pr(Correlation >= {}) \nwith {} pp, {} trials)".format(tau, npp, ntrials), fontweight = 'bold')
                axes1.set_title("Power = {}% \nbased on {} reps".format(np.round(power_estimate1*100, 2), nreps))
                axes1.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
                plt.tight_layout()
                fig2, axes2 = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output2["correlations"], label = "Correlations", ax = axes2, cut = 0)
                fig2.suptitle("Pr(Correlation >= {}) \nwith {} pp, {} trials)".format(tau, npp, ntrials), fontweight = 'bold')
                axes2.set_title("Power = {}% \nbased on {} reps".format(np.round(power_estimate2*100, 2), nreps))
                axes2.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
                plt.tight_layout()
        elif criterion == "GD":
            npp_pergroup = InputDictionary['npp_group'][row]
            npp = npp_pergroup*2
            meanLR1_g1, sdLR1_g1 = InputDictionary['meanLR1_g1'][row], InputDictionary['sdLR1_g1'][row]
            meanLR1_g2, sdLR1_g2 = InputDictionary['meanLR1_g2'][row], InputDictionary['sdLR1_g2'][row]
            meanLR2_g1, sdLR2_g1 = InputDictionary['meanLR2_g1'][row], InputDictionary['sdLR2_g1'][row]
            meanLR2_g2, sdLR2_g2 = InputDictionary['meanLR2_g2'][row], InputDictionary['sdLR2_g2'][row]
            corr = InputDictionary['corr'][row]
            meanInverseT_g1, sdInverseT_g1 = InputDictionary['meanInverseTemperature_g1'][row], InputDictionary['sdInverseTemperature_g1'][row]
            meanInverseT_g2, sdInverseT_g2 = InputDictionary['meanInverseTemperature_g2'][row], InputDictionary['sdInverseTemperature_g2'][row]
            typeIerror = InputDictionary['TypeIerror'][row]
            # Calculate tau based on the typeIerror and the df
            tau = -stat.t.ppf(typeIerror/2, npp-1)
            s_pooled1 = np.sqrt((sdLR1_g1**2 + sdLR1_g2**2) / 2)
            s_pooled2 = np.sqrt((sdLR2_g1**2 + sdLR2_g2**2) / 2)
            cohens_d1 = np.abs(meanLR1_g1-meanLR1_g2)/s_pooled1
            cohens_d2 = np.abs(meanLR2_g1-meanLR2_g2)/s_pooled2
            
            
            output1, power_estimate1, output2, power_estimate2 = power_estimation_groupdifference(npp_per_group = npp_pergroup, ntrials = ntrials,
                                               nreps = nreps, typeIerror = typeIerror, high_performance = full_speed,
                                               nreversals = nreversals, reward_probability = reward_probability,
                                               mean_LR1distributionG1 = meanLR1_g1, SD_LR1distributionG1 = sdLR1_g1,
                                               mean_LR1distributionG2 = meanLR1_g2, SD_LR1distributionG2=sdLR1_g2,
                                               mean_LR2distributionG1 = meanLR2_g1, SD_LR2distributionG1 = sdLR2_g1,
                                               mean_LR2distributionG2 = meanLR2_g2, SD_LR2distributionG2=sdLR2_g2, corr = corr,
                                               mean_inverseTempdistributionG1 = meanInverseT_g1, SD_inverseTempdistributionG1 = sdInverseT_g1,
                                               mean_inverseTempdistributionG2 = meanInverseT_g2, SD_inverseTempdistributionG2 = sdInverseT_g2)
            output1.to_csv(os.path.join(output_folder, 'OutputGD_LR1{}SD{}T{}R{}N{}REP{}D{}CORR{}REW.csv'.format(np.round(s_pooled1,2),
                                                                                            ntrials,
                                                                                            nreversals,
                                                                                            npp, nreps, np.round(cohens_d1,2),corr, reward_probability)))
            output2.to_csv(os.path.join(output_folder, 'OutputGD_LR2{}SD{}T{}R{}N{}REP{}D{}CORR{}REW.csv'.format(np.round(s_pooled2,2),
                                                                                        ntrials,
                                                                                        nreversals,
                                                                                        npp, nreps, np.round(cohens_d2,2),corr, reward_probability)))
            if HPC == False:
                fig1, axes1 = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output1["Statistic"], label = "T-statistic", ax = axes1)
                fig1.suptitle("Pr(T-statistic > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp_pergroup, ntrials), fontweight = 'bold')
                axes1.set_title("Power = {}% \nbased on {} reps with Cohen's d = {}".format(np.round(power_estimate1*100, 2), nreps, np.round(cohens_d1,2)))
                axes1.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
                fig2, axes2 = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output2["Statistic"], label = "T-statistic", ax = axes2)
                fig2.suptitle("Pr(T-statistic > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp_pergroup, ntrials), fontweight = 'bold')
                axes2.set_title("Power = {}% \nbased on {} reps with Cohen's d = {}".format(np.round(power_estimate2*100, 2), nreps, np.round(cohens_d2,2)))
                axes2.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
    
        elif criterion == "EC":
            npp = InputDictionary['npp'][row]
            meanLR1, sdLR1 = InputDictionary['meanLR1'][row], InputDictionary['sdLR1'][row]
            meanLR2, sdLR2 = InputDictionary['meanLR2'][row], InputDictionary['sdLR2'][row]
            meanInverseT, sdInverseT = InputDictionary['meanInverseTemperature'][row], InputDictionary['sdInverseTemperature'][row]
            True_correlation12 = InputDictionary['True_correlation12'][row]
            True_correlation13 = InputDictionary['True_correlation13'][row]
            True_correlation23 = InputDictionary['True_correlation23'][row]
            True_correlation = np.array([True_correlation12, True_correlation13, True_correlation23])
            typeIerror = InputDictionary['TypeIerror'][row]
            s_pooled1 = sdLR1
            s_pooled2 = sdLR2
            
            beta_distribution = stat.beta((npp/2)-1, (npp/2)-1, loc = -1, scale = 2)
            tau = -beta_distribution.ppf(typeIerror/2)
    
            output1, power_estimate1, output2, power_estimate2 = power_estimation_Excorrelation(npp = npp, ntrials = ntrials, nreps = nreps,
                                                                  typeIerror = typeIerror,
                                               high_performance = full_speed, nreversals = nreversals,
                                               reward_probability = reward_probability, mean_LR1distribution = meanLR1,
                                               SD_LR1distribution = sdLR1, mean_LR2distribution = meanLR2, SD_LR2distribution = sdLR2, mean_inverseTempdistribution = meanInverseT,
                                               SD_inverseTempdistribution = sdInverseT, True_correlation = True_correlation)
            output1.to_csv(os.path.join(output_folder, 'OutputEC_LR1{}SD{}TC{}T{}R{}N{}REP{}CORR{}REW.csv'.format(s_pooled1, True_correlation[1], ntrials,
                                                                                      nreversals,
                                                                                      npp, nreps, True_correlation12, reward_probability)))
            output2.to_csv(os.path.join(output_folder, 'OutputEC_LR2{}SD{}TC{}T{}R{}N{}REP{}CORR{}REW.csv'.format(s_pooled2, True_correlation[2], ntrials,
                                                                                  nreversals,
                                                                                  npp, nreps, True_correlation12, reward_probability)))
            if HPC == False:
                fig1, axes1 = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output1["Statistic"], label = "Correlation", ax = axes1, cut = 0)
                fig1.suptitle("Pr(Correlation > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp, ntrials), fontweight = 'bold')
                axes1.set_title("Power = {}% \nbased on {} reps with true correlation {}".format(np.round(power_estimate1*100, 2), nreps, True_correlation[1]))
                axes1.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
                fig2, axes2 = plt.subplots(nrows = 1, ncols = 1)
                sns.kdeplot(output2["Statistic"], label = "Correlation", ax = axes2, cut = 0)
                fig2.suptitle("Pr(Correlation > {}) \nconsidering a type I error of {} \nwith {} pp, {} trials".format(np.round(tau,2), typeIerror, npp, ntrials), fontweight = 'bold')
                axes2.set_title("Power = {}% \nbased on {} reps with true correlation {}".format(np.round(power_estimate2*100, 2), nreps, True_correlation[2]))
                axes2.axvline(x = tau, lw = 2, linestyle ="dashed", color ='k', label ='tau')
    
    
        else: print("Criterion not found")
        #final adaptations to the output figure & store the figure
        if HPC == False:
            fig1.legend(loc = 'center right')
            fig1.tight_layout()
            fig1.savefig(os.path.join(output_folder, 'Plot_LR1{}{}T{}R{}N{}M{}.jpg'.format(criterion,
                                                                                    np.round(s_pooled1, 2),
                                                                                    ntrials, nreversals,
                                                                                    npp, nreps)))
            fig2.legend(loc = 'center right')
            fig2.tight_layout()
            fig2.savefig(os.path.join(output_folder, 'Plot_LR2{}{}T{}R{}N{}M{}.jpg'.format(criterion,
                                                                                    np.round(s_pooled2, 2),
                                                                                    ntrials, nreversals,
                                                                                    npp, nreps)))
    
        # measure how long the power estimation lasted
        end_time = datetime.now()
        print("\nPower analysis ended at {}; run lasted {} hours.".format(end_time, end_time-start_time))
        power_estimate.append([power_estimate1, power_estimate2])
    temp = pd.DataFrame(power_estimate, columns = ['power1', 'power2'])
    temp.to_csv(os.path.join('./Compass_results', 'power_ic.csv'))

