import utils
import concurrent.futures
import numpy as np
import math
import time

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))
    return "calc cost: %d:%02d:%02d" % (h, m, s)

#*Searching for the best k and minimum tao_i
def main(T):
    list_, k_list = [], []
    for k_latent in np.linspace(0.1, 5, 60):
        print(f'step: {k_latent}')
        rao_model_ = utils.rao_model(mu_list, b, E, c, k, T, v, epsilon, u, k_latent)
        tao_k = rao_model_.ys(tf, q)
        list_.append(tao_k)
        k_list.append(k_latent)
        # clear_output(True)
        # try:
        #     plt.plot(list_)
        #     plt.show()
        # except:
        #     del list_[-1]

    return np.min(np.abs(list_)).evalf()/1e6, T

if __name__ == '__main__':
    start_ = time.time()

    mu_list = np.array([20, 38, 67, 161])*1e9 #*Shear modulus, Gpa -> Pa
    E = np.array([0.025, -0.074, -0.035, 0.059])*1.6022e-19 #*E_int for a particular atom specie, J, kg m^2 s^-2
    c = np.array([1/4, 1/4, 1/4, 1/4]) #*Atomic concentration

    b = 2.7886e-10 #*Burgers vector of the a/2[111] screw dislocation, m
    k = 8.617e-5 #*Boltzmann constant, in eV form
    k = 1.380649e-23 #*Standard form, m^2 kg s^-2 k^-1
    T = np.linspace(10, 293, 15) #*Temperature, K
    v = 5e12 #*Debye frequency, s^-1
    epsilon = 1e-3 #*Strain rate, s^-1
    u = 1 #*Unit convertion to Pa, kg m^-1 s^-2
    comment = 'MoNbTaW'
    max_work = 15 #*Maximum number of processors for parallel computing
    ys_list, t_list = [], []
    tf, q = 3.067, 2/3 #*Taylor factor for BCC and constant q.

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_work) as executor:
        ys_gen = executor.map(main, T)
        for ys_res in ys_gen:
            ys_list.append(ys_res[0])
            t_list.append(ys_res[1])

    second_to_hour(time.time() - start_)

    np.save(f'path', ys_list)
    np.save('path', t_list)
