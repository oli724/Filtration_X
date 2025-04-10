import numpy as np
nist_mu = np.array([  21.4785,  663.488,   293.3912])
exp_mu = np.array([ 23.1038, 703.57, 283.2843])
exp_err = np.array([ 2.5257, 73.76,21.9027])

demi_nist= np.log(2)/nist_mu
demi_exp = np.log(2)/exp_mu
err_rel = 100*((exp_mu-nist_mu)/nist_mu)
err_rel_demi = 100*((demi_exp-demi_nist)/demi_nist)
demi_err_exp = exp_err*(np.log(2)/(exp_mu**2))
print("diff rel mu:",err_rel)
print()
print("couche nist:",demi_nist*1e4)
print("couche exp:", demi_exp*1e4)
print("err_demi:",demi_err_exp*1e4)
print("rel diff demi:", err_rel_demi)