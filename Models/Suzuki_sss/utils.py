from sympy import *
import numpy as np
import math
import sympy
import matplotlib.pyplot as plt
from IPython.display import clear_output
from random import randrange

class rao_model():
    def __init__(self, mu, b, E, c, k, T, v, epsilon, u, k_, comment='None'):
        super(rao_model, self).__init__()
        self.mu = mu
        self.b = b
        self.E = E
        self.c = c
        self.k = k 
        self.T = T
        self.a = 0.942*b 
        self.lamda = 10*b 
        self.v = v
        self.epsilon = epsilon 
        self.u = u
        self.k_ = k_
        self.comment = comment

    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def f1(self):
        tao_chose_list = []
        for index in range(len(self.c)):
            mu_, c_, E_ = self.mu[index], self.c[index], self.E[index]
            tao_k = Symbol('tao_k', real=True)
            #*deltaV
            v_lefttop = 3*(self.k_**2)*(E_**2)*c_
            v_leftbot = 2*(tao_k**2)*self.a*(self.b**2)
            v_righttop = tao_k**2*(self.a**3)*(self.b**4)*(self.lamda**2)
            v_rightbot = 6*(self.k_**2)*(E_**2)*c_

            v_ = v_lefttop/v_leftbot + v_righttop/v_rightbot

            #*S
            s_lefttop = 18*(self.k_**2)*(E_**2)*c_*self.k*self.T
            s_leftbot = self.a**3*(self.b**4)*(self.lamda**2)
            s_righttop = (5*math.pi*self.k*self.T)**2*self.v*self.a*self.b
            s_rightbot = (mu_*self.b*v_)**2*self.epsilon

            s_ = s_lefttop/s_leftbot*sympy.log(s_righttop/s_rightbot)

            #*R
            r_top = 27*(self.k_**4)*(E_**4)*(c_**2)
            r_bot = (self.a**4)*(self.b**6)*(self.lamda**2)

            r_ = r_top/r_bot

            #*tao_k
            f1_ = tao_k**4 + s_*tao_k - r_bot
            res_list, tao_list = [], []
            tao_k_c = 2e8
            temp = 1e5
            speed = 1e9
            step = 0

            while True:
                f1_res = abs(f1_.subs(tao_k, tao_k_c))
                res_list.append(abs(f1_res.evalf()))
                tao_list.append(tao_k_c)
                tao_chose = tao_k_c
                tao_chose += (np.random.rand()-0.5)*speed
                f1_res_ = abs(f1_.subs(tao_k, tao_chose))
                res = (f1_res - f1_res_).evalf()

                accep = np.random.rand()
                prob_ = np.min([1, exp(res/temp)])
                if accep <= prob_:
                    tao_k_c = tao_chose

                temp /= 1.1
                speed /= 1.2

                # if step % 10 == 0:
                    # clear_output(True)

                    # fig, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.plot(res_list)
                    # ax1.set_ylim([1e7, 1e9])
                    # ax2.plot(tao_list)
                    # plt.suptitle(f'Num steps: {step}, Minimum res: {f1_res}\n{tao_chose/1e6} MPa')
                    # plt.suptitle(f'Num steps: {step}, Minimum res: {np.min(res_list)}\n{tao_chose/1e6} MPa')
                    # plt.show()

                step += 1
                if step >= 888:
                    tao_chose_list.append(tao_chose)
                    break
                
        return np.array(tao_chose_list)

    def f4(self):
        j_list = []
        for index in range(len(self.c)):
            mu_, c_, E_ = self.mu[index], self.c[index], self.E[index]
            x = Symbol('x')
            inte_ = sympy.exp(-x**2/2)
            f4_ = 1/sympy.sqrt(2*math.pi)*integrate(inte_, (x, self.k_, oo))
            L_ = self.b/(f4_*3*c_)
            j_ = mu_*self.b/(4*L_)
            j_list.append(j_.evalf())

        return np.array(j_list)

    def tao(self):
        return (self.f1()+self.f4())*self.u

    def ys(self, tf, q):
        tao_list = self.tao()
        ys_ = np.sum(tao_list**(1/q))**q
        if self.comment != 'None':
            print(self.comment)
        return ys_.evalf()*tf