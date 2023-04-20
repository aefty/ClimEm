import numpy as np
import pickle
import matplotlib, matplotlib.pyplot as plt

##########################
# Example simulation function for CO2 and temperature
##########################
#A    : Operator such that dm/dt = Am + e
#m_eq : Equilibrium mass (GtC)
#m_0  : Initial starting value of masses
#T    : Length of simulation
#e    : Emissions in GtC, where e[t] is the emission at time index t (not len(e)==T)"
def simulate(A, m_eq, m_0, T, e=None, 
    c1=0.137, 
    c3=0.73, 
    c4=0.00689, 
    F2XCO2=3.45,
    T2XCO2=3.25, 
    TAT0=0.0,
    TOC0=0.0,
    delta_t=1
    ) -> list:

    # Initial variables
    if e is None:
        e = np.zeros(T)
    else:
        assert T == len(e)

    m = np.empty(shape=(A.shape[0], T))
    temprature_oc = np.zeros(T)
    temprature_at = np.zeros(T)

    B = A + np.eye(A.shape[0])
    C = np.linalg.inv(np.eye(A.shape[0]) - A)

    temprature_at[0] = TAT0
    temprature_oc[0] = TOC0

    m[:, 0] = m_0
    m[0, 0] += e[0]
    for t in range(1, T):
        m[:, t]  = C @ m[:, t - 1]
        m[0, t] += e[t]

        F_t = (1+0.3) * F2XCO2 * np.log(m[0, t-1] / m_eq[0]) / np.log(2) 

        delta_temp       = temprature_at[t - 1] - temprature_oc[t - 1]
        temprature_at[t] = temprature_at[t - 1] + delta_t * c1 * (F_t - F2XCO2/T2XCO2*temprature_at[t-1] - c3*(delta_temp))
        temprature_oc[t] = temprature_oc[t - 1] + delta_t * c4 * (delta_temp)

    return [m, temprature_at, temprature_oc]


# open a file, where you stored the pickled data
file = open('data_dump.pkl', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()

##########################
# We have three model configuation
##########################
# 3SR three reservoirs connected in serial
# AT -> O_1 -> [O_2]
#
# 4PR four reservoirs connected in parallel
#  - -> [L_1]
# |
# AT -> O_1 -> [O_2]  
#
# 5PR five reservoirs connected in parallel
#  - -> L_1 - [L_2]
# |
# AT -> O_1 -> [O_2]  

##########################
# We have three test types
##########################
# MMMU (index 0): MMM + 2SD
# MMM  (index 1): Multi modal mean
# MMMD (index 2): MMM - 2SD

#print(data['3SR'].keys())

model_name_set = ['3SR','4PR','5PR']
test_name_set  = ['MMMU','MMM','MMMD']

spec = 'a'
for model_name in model_name_set:

    temp = np.zeros( shape=(len(data[model_name][spec][0]),3))

    for test_type_inx, test_type_name in enumerate(test_name_set):

        # operator
        
        print(model_name,":",test_type_name,':',spec,':\n',data[model_name][spec][test_type_inx]*100)

        temp[:,test_type_inx] = data[model_name][spec][test_type_inx]*100
        # m_eq is shared among each model configuration
        #spec = 'm_eq'
        #print(model_name,":",test_type_name,':',spec,':\n',data[model_name][spec][test_type_inx])

        #spec = 'm_ini'
        #print(model_name,":",test_type_name,':',spec,':\n',data[model_name][spec][test_type_inx])

        # Year at which we reach 2.12*421 ppm starting from 1765 ( this is date corresponding to m_ini)
        #spec = 't_current'
        #print(model_name,":",test_type_name,':',spec,':\n',data[model_name][spec][test_type_inx])


    print(np.round(temp,2),'\n')
if 0:
    ##########################
    # example simlation 
    ##########################
    # 50GtC Pulse from Equlibrum for 5PR model using MMM tunning 
    A    = data['5PR']['A'][1]
    m_eq = data['5PR']['m_eq'][1]
    T    = 1000
    e    =[0]*T
    e[0] =50
    [m, temprature_at, temprature_oc] = simulate(A=A,m_eq=m_eq,m_0=m_eq,T=T,e=e)

    # ordering of the indxes bellow applied to all modesl
    plt.plot(m[0,:]-m_eq[0],label='ATM')
    plt.plot(m[1,:]-m_eq[1],label='O1')
    plt.plot(m[2,:]-m_eq[2],label='O2')
    plt.plot(m[3,:]-m_eq[3],label='L1')
    plt.plot(m[4,:]-m_eq[4],label='L2')
    plt.legend()
    plt.show()