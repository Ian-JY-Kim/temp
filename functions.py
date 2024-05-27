import numpy as np
from parameter import *
from scipy.optimize import fsolve
import math
from scipy.optimize import minimize

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
vec_monopoly = np.vectorize(monopoly)

def duopoly(x_1, x_2, w_1, w_2, xi_1, xi_2, omega_1, omega_2):
    try:
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
    
    except:
        return 100, 100, 100, 100, 100, 100
vec_duopoly = np.vectorize(duopoly)