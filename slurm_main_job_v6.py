import sys
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import random
from scipy.optimize import fsolve
from scipy.optimize import minimize
import math
from scipy.stats import norm
from statistics import NormalDist
from scipy.integrate import quad
from itertools import product
from parameter import *


# get the job number from the slurm
job_num_seed = int(sys.argv[1])
num_duty_per_job = 2

end_num = job_num_seed * num_duty_per_job
start_num = end_num - (num_duty_per_job - 1)

MC_master_data = pd.read_csv("MC_master_data.csv")
MC_master_data = MC_master_data[['x_1', 'x_2', 'w_1', 'w_2', 'job_array_num']]




job_num_list = []
xi_1_list    = []
xi_2_list    = []


# work
for job_num in range(start_num, end_num+1):

    # get the working data    
    sample_data = MC_master_data.loc[MC_master_data['job_array_num'] == job_num]
    # get the Z information
    x_1, x_2, w_1, w_2 = sample_data.values[0][0:4]


    def monopoly(x, w, xi, omega):
        mc = np.exp(gamma*w + omega)
        T = np.exp(beta*x - alpha*mc + xi)

        #(1) find Y
        def monopoly_eqn(var):
            Y = var
            eq = 1 - Y + T*np.exp(-Y)
            return eq
        Y = fsolve(monopoly_eqn, 1)[0]
        
        pi = (1/alpha)*(Y-1) 
        price = Y/alpha + mc
        share = pi/(price-mc)

        return pi, price, share

    def duopoly(x_1, x_2, w_1, w_2, xi_1, xi_2, omega_1, omega_2):
        mc_1 = np.exp(gamma*w_1 + omega_1) 
        mc_2 = np.exp(gamma*w_2 + omega_2)
        T_1 = np.exp(beta*x_1 - alpha*mc_1 + xi_1)
        T_2 = np.exp(beta*x_2 - alpha*mc_2 + xi_2)
        
        def duopoly_fun(Y):
            Y_1, Y_2 = Y
            eqn1 = Y_1 - math.log(T_1*(Y_2-1)) + math.log(1-Y_2+T_2*np.exp(-Y_2))        
            return abs(eqn1)
        
        def c1(Y):
            'Y_1 exp term greater than 0'
            Y_1, Y_2 = Y
            return 1-Y_1+T_1*np.exp(-Y_1)

        def c2(Y):
            'Y_2 exp term greater than 0'
            Y_1, Y_2 = Y 
            return 1-Y_2+T_2*np.exp(-Y_2)
        
        def c3(Y):
            Y_1, Y_2 = Y
            return Y_2 - math.log(T_2*(Y_1-1)) + math.log(1-Y_1+T_1*np.exp(-Y_1))

        bnds = ((1.000001, None), (1.000001, None))
        cons = ({'type': 'ineq', 'fun': c1}, 
                {'type': 'ineq', 'fun': c2},
                {'type': 'eq', 'fun': c3})
        initial_point = (1.0001, 1.0001)
        res = minimize(duopoly_fun, initial_point, method = 'SLSQP', bounds=bnds, constraints=cons)
        Y_1 = res.x[0]
        Y_2 = res.x[1]
        
        pi_1 = (1/alpha)*(Y_1-1)
        pi_2 = (1/alpha)*(Y_2-1)

        price_1 = Y_1/alpha + mc_1
        price_2 = Y_2/alpha + mc_2

        share_1 = pi_1/(price_1 - mc_1)
        share_2 = pi_2/(price_2 - mc_2)

        return pi_1, pi_2, price_1, price_2, share_1, share_2

    def EXP_pi_mono(x, w, xi):
        #exogenous cost shock probability
        prob_good = 1/2
        prob_bad = 1/2
        #calculate monopoly profit
        pi_mono_good = monopoly(x, w, xi, -0.05)[0]
        pi_mono_bad  = monopoly(x, w, xi,  0.35)[0]
        #return expected monopoly profit
        return prob_good*pi_mono_good + prob_bad*pi_mono_bad

    upper_bound = 6
    num_discretize = 500
    def EXP_pi_duo_f1_trapz(x_1, x_2, w_1, w_2, xi_1, xi_2_star):
        xi_2_vec = np.linspace(xi_2_star, upper_bound + mu_xi_2, num_discretize)
        vec_duo = np.vectorize(duopoly)
        trapz_fun1 = vec_duo(x_1, x_2, w_1, w_2, xi_1, xi_2_vec,-0.05, 0.35)[0] * norm.pdf(xi_2_vec, mu_xi_2, sigma_xi_2)
        trapz_fun2 = vec_duo(x_1, x_2, w_1, w_2, xi_1, xi_2_vec,-0.05,-0.05)[0] * norm.pdf(xi_2_vec, mu_xi_2, sigma_xi_2)
        trapz_fun3 = vec_duo(x_1, x_2, w_1, w_2, xi_1, xi_2_vec, 0.35, 0.35)[0] * norm.pdf(xi_2_vec, mu_xi_2, sigma_xi_2)
        trapz_fun4 = vec_duo(x_1, x_2, w_1, w_2, xi_1, xi_2_vec, 0.35,-0.05)[0] * norm.pdf(xi_2_vec, mu_xi_2, sigma_xi_2)
        
        trapz_sum = 0
        for i in range(1,num_discretize):
            delta_x = xi_2_vec[i] - xi_2_vec[i-1]
            height1 = (1/2)*(trapz_fun1[i] + trapz_fun1[i-1])
            height2 = (1/2)*(trapz_fun2[i] + trapz_fun2[i-1])
            height3 = (1/2)*(trapz_fun3[i] + trapz_fun3[i-1])
            height4 = (1/2)*(trapz_fun4[i] + trapz_fun4[i-1])
            height = (1/4)*(height1+height2+height3+height4)
            trapz_sum += delta_x * height
        prob_f2_in  = 1 - norm.cdf(xi_2_star, loc = mu_xi_2, scale = sigma_xi_2)
        return (1/prob_f2_in)*trapz_sum 

    def EXP_pi_duo_f2_trapz(x_1, x_2, w_1, w_2, xi_1_star, xi_2):
        xi_1_vec = np.linspace(xi_1_star, upper_bound + mu_xi_1, num_discretize)
        vec_duo = np.vectorize(duopoly)
        trapz_fun1 = vec_duo(x_1, x_2, w_1, w_2, xi_1_vec, xi_2,-0.05, 0.35)[1] * norm.pdf(xi_1_vec, mu_xi_1, sigma_xi_1)
        trapz_fun2 = vec_duo(x_1, x_2, w_1, w_2, xi_1_vec, xi_2,-0.05,-0.05)[1] * norm.pdf(xi_1_vec, mu_xi_1, sigma_xi_1)
        trapz_fun3 = vec_duo(x_1, x_2, w_1, w_2, xi_1_vec, xi_2, 0.35, 0.35)[1] * norm.pdf(xi_1_vec, mu_xi_1, sigma_xi_1)
        trapz_fun4 = vec_duo(x_1, x_2, w_1, w_2, xi_1_vec, xi_2, 0.35,-0.05)[1] * norm.pdf(xi_1_vec, mu_xi_1, sigma_xi_1)
        
        trapz_sum = 0
        for i in range(1,num_discretize):
            delta_x = xi_1_vec[i] - xi_1_vec[i-1]
            height1 = (1/2)*(trapz_fun1[i] + trapz_fun1[i-1])
            height2 = (1/2)*(trapz_fun2[i] + trapz_fun2[i-1])
            height3 = (1/2)*(trapz_fun3[i] + trapz_fun3[i-1])
            height4 = (1/2)*(trapz_fun4[i] + trapz_fun4[i-1])
            height = (1/4)*(height1+height2+height3+height4)
            trapz_sum += delta_x * height
        prob_f1_in  = 1 - norm.cdf(xi_1_star, loc = mu_xi_1, scale = sigma_xi_1)
        return (1/prob_f1_in)*trapz_sum 



    def Z_1(xi_1_star, xi_2_star, x_1, x_2, w_1, w_2):
        # distribution of xi_2
        mu = mu_xi_2
        sigma = sigma_xi_2

        # calculate cdf in needs
        prob_f2_out = norm.cdf(xi_2_star, loc = mu, scale = sigma)
        prob_f2_in  = 1 - prob_f2_out

        # calculate expected pi according to the market structure
        pi_mono_f1 = EXP_pi_mono(x_1, w_1, xi_1_star)
        pi_duo_f1  = EXP_pi_duo_f1_trapz(x_1, x_2, w_1, w_2, xi_1_star, xi_2_star)

        # return indifference condition
        return prob_f2_out*pi_mono_f1 + prob_f2_in*pi_duo_f1 - kappa_1

    def Z_2(xi_1_star, xi_2_star, x_1, x_2, w_1, w_2):
        # distribution of xi_1
        mu = mu_xi_1
        sigma = sigma_xi_1

        # calculate cdf in needs
        prob_f1_out = norm.cdf(xi_1_star, loc = mu, scale = sigma)
        prob_f1_in  = 1 - prob_f1_out

        # calculate expected pi according to the market structure
        pi_mono_f2 = EXP_pi_mono(x_2, w_2, xi_2_star)
        pi_duo_f2  = EXP_pi_duo_f2_trapz(x_1, x_2, w_1, w_2, xi_1_star, xi_2_star)

        # return indifference condition
        return prob_f1_out*pi_mono_f2 + prob_f1_in*pi_duo_f2 - kappa_2


    def min_obj(theta, Z):
        xi_1_star, xi_2_star = theta
        x_1, x_2, w_1, w_2 = Z
        Z_1_value = Z_1(xi_1_star, xi_2_star, x_1, x_2, w_1, w_2)
        Z_2_value = Z_2(xi_1_star, xi_2_star, x_1, x_2, w_1, w_2)
        return Z_1_value**2 + Z_2_value**2


    try:
        initial_point = [0.8, 0.8]
        bnds = ((-6, 6), (-6, 6))
        res = minimize(min_obj, initial_point, args = sample_data.values[0][0:4], method="Nelder-Mead", bounds=bnds, options = {'maxiter': 300})

        # save the result
        job_num_list.append(job_num)
        xi_1_list.append(res.x[0])
        xi_2_list.append(res.x[1])


    except: 
        job_num_list.append(job_num)
        xi_1_list.append(100)
        xi_2_list.append(100)




# gen df and export csv 
df_out = df({"job_num": job_num_list,
             "xi_1_star": xi_1_list, 
             "xi_2_list": xi_2_list})

file_name = "result_" + str(job_num_seed) + ".csv"
df_out.to_csv("./result/" + file_name)