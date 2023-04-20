import numpy as np
import pickle
from prettytable import PrettyTable
import lib_data_load as data_load
from lib_models import model_3sr, model_4pr, model_5pr


def l2_err(x, x_true):
    temp = (x - x_true) / (x_true + 1e-12)
    return np.linalg.norm(temp, ord=2) / len(x)


def l1_err(x, x_true):
    temp = (x - x_true) / (x_true + 1e-12)
    return np.linalg.norm(temp, ord=1) / len(x)


def linf_err(x, x_true):
    temp = (x - x_true) / (x_true + 1e-12)
    return np.max(np.abs(temp))


def lw_err(x, x_true):
    temp = np.abs((x - x_true) / (x_true + 1e-12))

    k = np.log(1 / 2) / (len(x) - 1)
    w = np.exp(np.array(range(0, len(x))) * k)

    return np.dot(w, temp) / len(x)


def eig(A: np.array) -> list:
    """eigen value decomposion

    Args:
        A (np.array): Matrix

    Returns:
        list: [eig_val,eig_vec] 
        normalized (unit “length”) eigenvectors, such that the column eig_vec[:,i] 
        is the eigenvector corresponding to the eigenvalue eig_val[i]. 
        Sorted biggest to absval (largest to smallest)
    """
    [eig_val, eig_vec] = np.linalg.eig(A)

    idx = np.abs(eig_val).argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    return [eig_val, eig_vec]


def simulate_pulse(A: np.array, m0: np.array, T: int) -> np.array:
    """Simulate the ODE for the given operator starting from state x0 for T steps

    Args:
        A (np.array): Matrix operator
        T (int): Number of time steps
        x0 (np.array): Initial state

    Returns:
        np.array: Simulated path from x0 for T steps 
    """

    [eig_val, eig_vec] = np.linalg.eig(A)

    # idx = np.abs(eig_val).argsort()[::-1]
    # eig_val = eig_val[idx]
    # eig_vec = eig_vec[:,idx]

    # Initial variables
    p = len(m0)
    m = np.zeros(shape=(p, T))

    # solve system Vc=x0 where V is eigen vectors
    c = np.linalg.solve(eig_vec, m0)

    # E[i,j] = exp(\lambda_i x t_j)
    E = np.exp(np.outer(eig_val, np.array(range(0, T))))

    for i in range(0, p):
        m += c[i] * np.outer(eig_vec[:, i], E[i, :])

    return m



def test_temp(m):

    c1       =0.137
    c3       =0.73 
    c4       =0.00689  
    F2XCO2   =3.45
    T2XCO2   =3.25 
    TAT0     =0
    TOC0     =0
    delta_t  =1



    # this is the difference before 604!!
    m_eq_at = 589

    T = len(m)

    temprature_at = np.zeros(T)
    temprature_oc = np.zeros(T)

    temprature_at[0] =  TAT0
    temprature_oc[0] =  TOC0

    for t in range(1,T):

        F_t = 1.3 * F2XCO2 * np.log(m[t-1] / m_eq_at) / np.log(2) 

        delta_temp       = temprature_at[t - 1] - temprature_oc[t - 1]
        temprature_at[t] = temprature_at[t - 1] + delta_t * c1 * (F_t - F2XCO2/T2XCO2*temprature_at[t-1] - c3*(delta_temp))
        temprature_oc[t] = temprature_oc[t - 1] + delta_t * c4 * (delta_temp)

    return [temprature_at, temprature_oc]


def simulate(A, m_eq, m0, T, e=None, 
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

    m = np.empty(shape=(A.shape[0], T))
    temprature_oc = np.zeros(T)
    temprature_at = np.zeros(T)

    B = A + np.eye(A.shape[0])
    C = np.linalg.inv(np.eye(A.shape[0]) - A)


    temprature_at[0] = TAT0
    temprature_oc[0] = TOC0

    m[:, 0] = m0
    m[0, 0] += e[0]
    for t in range(1, T):
        m[:, t]  = C @ m[:, t - 1]
        m[0, t] += e[t]

        F_t = (1+0.3) * F2XCO2 * np.log(m[0, t-1] / m_eq[0]) / np.log(2) 

        delta_temp       = temprature_at[t - 1] - temprature_oc[t - 1]
        temprature_at[t] = temprature_at[t - 1] + delta_t * c1 * (F_t - F2XCO2/T2XCO2*temprature_at[t-1] - c3*(delta_temp))
        temprature_oc[t] = temprature_oc[t - 1] + delta_t * c4 * (delta_temp)

    return [m, temprature_at, temprature_oc]


def load_results(root, test_name, model_set=[model_3sr, model_4pr, model_5pr], T_sim=300, multiplier=1):
    results = {}

    test_type_set = test_name.split('-')[0].split('_')
    rho = test_name.split('-')[1].split('_')
    rho = np.array([float(rho_i) for rho_i in rho])

    k = len(test_type_set)

    for model_inx, model in enumerate(model_set):

        [a_size, m_size, info] = model()
        model_name = info['name']
        folder = root + test_name + '/' + model_name + '/result.pkl'

        with open(folder, 'rb') as f:
            data = pickle.load(f)

        results[model_name] = {}
        results[model_name]['rho'] = rho
        results[model_name]['test_type_set'] = test_type_set + ['AVG']
        results[model_name]['model'] = model

        results[model_name]['A']    = [0] * (k+1)
        results[model_name]['m_eq'] = [0] * (k+1)  # equlibrium masses
        results[model_name]['a']    = [0] * (k+1)  # values of a in A

        results[model_name]['m_sim']       = [0] * (k+1)  # 100 GtC pulse simulation
        results[model_name]['m_benchmark'] = [0] * (k+1)  # 100 GtC pulse benchmark

        results[model_name]['time_scale'] = [0] * (k+1)

        results[model_name]['err_l2_50']   = [0] * (k+1)
        results[model_name]['err_l2_500']  = [0] * (k+1)
        results[model_name]['err_l2_1000'] = [0] * (k+1)

        results[model_name]['err_l1_50']   = [0] * (k+1)
        results[model_name]['err_l1_500']  = [0] * (k+1)
        results[model_name]['err_l1_1000'] = [0] * (k+1)
        results[model_name]['dif_t50_l2']  = [0] * (k+1)
        results[model_name]['dif_t50_l1']  = [0] * (k+1)
        results[model_name]['dif_t50_pr']  = [0] * (k+1)

        results[model_name]['max_flux'] = [0] * (k+1)
        results[model_name]['max_disp'] = [0] * (k+1)

        results[model_name]['abs_del_a'] = [0] * (k+1)
        results[model_name]['time_scale_mean']  = [0] * (k+1)

         # identify ocean and land reservoirs
        inx_ocean = []
        inx_land  = []

        for m_i_name in info['m_names']:
            if 'O_' in m_i_name:
                inx_ocean.append(info['m_names'].index(m_i_name))
            elif 'L_' in m_i_name:
                inx_land.append(info['m_names'].index(m_i_name))
                
        for test_type_inx, test_type_name in enumerate(test_type_set):
                
            ############################################################
            # definition of the model
            ############################################################
            A    = data['A'][test_type_inx]
            m_eq = data['m_eq'][test_type_inx]
            x_vec = data['x_vec'][test_type_inx]

            [eig_val, eig_vec] = eig(A)
            eig_val_trim = eig_val[np.abs(eig_val) > 1e-12]
            time_scale = 1.0 / np.abs(eig_val_trim)

            results[model_name]['A'][test_type_inx] = A
            results[model_name]['m_eq'][test_type_inx] = np.round(m_eq, 0)
            results[model_name]['a'][test_type_inx] = np.round(x_vec[0:a_size], 6)
            results[model_name]['time_scale'][test_type_inx] = time_scale

            results[model_name]['time_scale_mean'][test_type_inx] = np.mean(time_scale)

             ############################################################
            # error metrics for 100GtC pulse at T=50, 500 and 1000
            ############################################################
            p = A.shape[0]
            m0 = np.zeros(p)
            m0[0] = 100

            m_sim = simulate_pulse(A=A, m0=m0, T=T_sim)
            
            [m_benchmark,_] = data_load.pulse_fraction(test_type=test_type_name, T=T_sim) 

            m_benchmark=m_benchmark* 100

            results[model_name]['m_sim'][test_type_inx] = m_sim
            results[model_name]['m_benchmark'][test_type_inx] = m_benchmark

            T = 50
            err_l2_50 = l2_err(m_sim[0, 0:T], m_benchmark[0:T])
            err_l1_50 = l1_err(m_sim[0, 0:T], m_benchmark[0:T])

            T = 500
            err_l2_500 = l2_err(m_sim[0, 0:T], m_benchmark[0:T])
            err_l1_500 = l1_err(m_sim[0, 0:T], m_benchmark[0:T])

            T = 1000
            err_l2_1000 = l2_err(m_sim[0, 0:T], m_benchmark[0:T])
            err_l1_1000 = l1_err(m_sim[0, 0:T], m_benchmark[0:T])

            results[model_name]['err_l2_50'][test_type_inx] = err_l2_50 * multiplier
            results[model_name]['err_l2_500'][test_type_inx] = err_l2_500 * multiplier
            results[model_name]['err_l2_1000'][test_type_inx] = err_l2_1000 * multiplier

            results[model_name]['err_l1_50'][test_type_inx] = err_l1_50 * multiplier
            results[model_name]['err_l1_500'][test_type_inx] = err_l1_500 * multiplier
            results[model_name]['err_l1_1000'][test_type_inx] = err_l1_1000 * multiplier

            ############################################################
            # compute difference betwee carbon absorved by oceans and lands at t=50
            ############################################################

             # we done have oceans/land ignore not relevant
            if len(inx_ocean) == 0 or len(inx_land) == 0:
                dif_t50_l2 = 0
                dif_t50_l1 = 0
            else:
                T = 50
                m_T = m_sim[:, T]
                temp = [np.sum(m_T[inx_ocean]), np.sum(m_T[inx_land])]
                ratio = np.diff(temp) / (np.sum(temp) + 1e-12)
                dif_t50_l2 = np.linalg.norm(ratio)
                dif_t50_l1 = np.abs(ratio)

            results[model_name]['dif_t50_l2'][test_type_inx] = dif_t50_l2 * multiplier
            results[model_name]['dif_t50_l1'][test_type_inx] = dif_t50_l1 * multiplier
            results[model_name]['dif_t50_pr'][test_type_inx] = dif_t50_l1 * 100

             ############################################################
            # max flux and displacment
            ############################################################
            max_flux = np.linalg.norm(A - A.T)
            max_disp = np.diag(np.abs(A)).max()

            results[model_name]['max_flux'][test_type_inx] = max_flux * multiplier
            results[model_name]['max_disp'][test_type_inx] = max_disp * multiplier


        # compute values for abs_del_a ... we need to looop again
        for test_type_inx, test_type_name in enumerate(test_type_set):
            results[model_name]['abs_del_a'][test_type_inx]  = np.linalg.norm( results[model_name]['a'][0] -  results[model_name]['a'][test_type_inx]  )
        
        # Compute averages for errors
        for key in results[model_name]:

 
            if key in ['err_l2_50', 'err_l2_500', 'err_l2_1000', 'err_l1_50', 'err_l1_500', 'err_l1_1000', 'dif_t50_l2','dif_t50_l2','dif_t50_pr', 'max_flux', 'max_disp','abs_del_a','time_scale_mean']:
                results[model_name][key][-1] = np.sum(results[model_name][key]) / k

    return results


def tabulate(results, 
    title='Table', 
    vars=['a', 'm_eq', 'time_scale', 'err_l2_50', 'err_l2_500', 'err_l2_1000', 'err_l1_50', 'err_l1_500', 'err_l1_1000', 'dif_t50_l2', 'max_flux', 'max_disp','abs_del_a','time_scale_mean'], 
    test_type_show=['MMMU', 'MMM', 'MMMD']
    ):

    table = PrettyTable()
    table.field_names = ['model', 'type', 'rho'] + vars
    table.float_format = "0.3"
    table.title = title

    data = {}
    data['key'] = table.field_names
    data['val'] = []

    for model_inx, model_name in enumerate(results.keys()):
        for test_type_inx, test_type_name in enumerate(results[model_name]['test_type_set']):
            row = []
            if test_type_name in test_type_show:
                row += [model_name]
                row += [test_type_name]
                row += [results[model_name]['rho']]
                temp = []
                for u in table.field_names[3:]:
                    temp += [results[model_name][u][test_type_inx]]
                row += temp
                table.add_row(row)
                data['val'].append(temp)

    print(table)