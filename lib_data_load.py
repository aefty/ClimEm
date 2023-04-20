import numpy as np
import netCDF4 as nc
import glob


#benchmark: 100 GtC to atmosphere in 2015, Joos et al. (2013)
#see https://acp.copernicus.org/articles/13/2793/2013/acp-13-2793-2013-supplement.pdf
def pulse_fraction_old(test_type, T):

    if test_type == 'MESMO' or test_type == 0:
        param_a_IRFCO2 = [2.848E-01, 2.938E-01, 2.382E-01, 1.831E-01]
        param_tau_IRFCO2 = [4.543E+02, 2.500E+01, 2.014E+00]
    elif test_type == 'MMM' or test_type == 1:
        param_a_IRFCO2 = [2.173E-01, 2.240E-01, 2.824E-01, 2.763E-01]
        param_tau_IRFCO2 = [3.944E+02, 3.654E+01, 4.304E+00]
    elif test_type == 'LOVECLIM' or test_type == 2:
        param_a_IRFCO2 = [8.539E-08, 3.606E-01, 4.503E-01, 1.891E-01]
        param_tau_IRFCO2 = [1.596E+03, 2.171E+01, 2.281E+00]
    else:
        raise Exception("Invalid test_type")

    pulse_frac = np.empty(T)

    a = param_a_IRFCO2
    tau = param_tau_IRFCO2

    for t in range(0, T):
        pulse_frac[t] = a[0] + a[1] * np.exp(-t / tau[0]) + a[2] * np.exp(-t / tau[1]) + a[3] * np.exp(-t / tau[2])

    return pulse_frac


def pulse_fraction(test_type, T,folder_name: str = 'data/pulse/'):

    data = np.loadtxt(folder_name+'IRF_PI100_SMOOTHED_CO2.dat') 

    name=''

    if test_type == 'NCAR':
        pulse_frac = data[:,1]
        name = 'NCAR CSM1.4'
    elif test_type == 'BERN3D':
        pulse_frac = data[:,2]
        name = 'Bern3D-LPJ'
    elif test_type == 'BERN25D':
        pulse_frac = data[:,3]
        name = 'Bern2.5D-LPJ'
    elif test_type == 'CLIMBER2':
        pulse_frac = data[:,4]
        name = 'CLIMBER2-LPJ'
    elif test_type == 'DCESS':
        pulse_frac = data[:,5]
        name = 'DCESS'
    elif test_type == 'GENIE':
        pulse_frac = data[:,6]
        name = 'GENIE '
        # (ensemble median)
    elif test_type == 'LOVECLIM':
        pulse_frac = data[:,7]
        name = 'LOVECLIM'
    elif test_type == 'MESMO':
        pulse_frac = data[:,8]
        name = 'MESMO'
    elif test_type == 'UVIC29':
        pulse_frac = data[:,9]
        name = 'UVic2.9'
    elif test_type == 'BERNSAR':
        pulse_frac = data[:,10]
        name = 'Bern-SAR'
    elif test_type == 'MMM':
        pulse_frac = data[:,11]
        name = '$\mu$'
    elif test_type == 'MMMU':
        pulse_frac = data[:,11] + data[:,12]*2
        name = '$\mu^+$'
    elif test_type == 'MMMD':
        pulse_frac = data[:,11] - data[:,12]*2
        name = '$\mu^-$'
    else:
        raise Exception("Invalid test_type")
    

    pulse_frac = pulse_frac[pulse_frac<1e10]

    #convert to GTC from concentraitonst
    pulse_frac = pulse_frac *2.12

    # add 100 GtC in the first index
    pulse_frac = np.append([100],pulse_frac) 

    #scale everyting back to %
    pulse_frac = pulse_frac/100

    pulse_frac = pulse_frac[0:T]

    return [pulse_frac,name] 





def cmip_emission(scenerio_name: str, T_start: int = 1765, T_end: int = 2500, folder_name: str = 'data/emission/'):

    assert T_start >= 1765
    assert T_end <= 2500

    temp = np.loadtxt(folder_name + scenerio_name + '.txt')
    data_val = np.array(np.sum(temp[:, 1:], 1), dtype='float')
    data_year = np.array(temp[:, 0], dtype='int')

    # shortcut for finding index of given value
    inx_find = lambda x, val: int(np.where(x == val)[0])
    i_start = inx_find(data_year, int(T_start))
    i_end = inx_find(data_year, int(T_end)) + 1

    data_val = data_val[i_start:i_end]
    data_year = data_year[i_start:i_end]

    return [data_val, data_year]


def cmip_concentration(scenerio_name: str, T_start: int = 1765, T_end: int = 2500, folder_name: str = 'data/concentration/'):

    assert T_start >= 1765
    assert T_end <= 2500

    temp = np.loadtxt(folder_name + scenerio_name + '.txt')
    data_val = np.array(temp[:, 1], dtype='float')
    data_year = np.array(temp[:, 0], dtype='int')

    # shortcut for finding index of given value
    inx_find = lambda x, val: int(np.where(x == val)[0])

    i_start = inx_find(data_year, int(T_start))
    i_end = inx_find(data_year, int(T_end)) + 1

    data_val = data_val[i_start:i_end]
    data_year = data_year[i_start:i_end]

    return [data_val, data_year]


def cmip_temperature(scenerio_name: str, T_start: int = 1765, T_end: int = 2500, folder_name: str = 'data/temperature/'):

    assert T_start >= 1765
    assert T_end <= 2500

    year_min = 1765
    year_max = 2500

    file_name_set = glob.glob(folder_name + scenerio_name + '/' + '*_am_*.nc')

    year_set = []
    val_set = []

    t_min =[]

    for file_name in file_name_set:
        data = nc.Dataset(file_name)

        year_set += [np.array(np.array(data['time']).flatten() / 1e4, dtype='int')]

        temp =  np.array(data['ts']).flatten()
        temp =  temp - np.mean(temp[0:5])

        val_set  += [temp]

    data_val_full = np.zeros(shape=(len(val_set), year_max - year_min + 1))
    data_year_full = np.array(range(year_min, year_max + 1))

    #print(data_year_full)

    # shortcut for finding index of given value
    inx_find = lambda x, val: int(np.where(x == val)[0])

    for i in range(0, len(data_val_full)):

        i_start = inx_find(data_year_full, min(year_set[i]))
        i_end   = inx_find(data_year_full, max(year_set[i])) + 1

        data_val_full[i, i_start:i_end] = val_set[i]

    i_start = inx_find(data_year_full, int(T_start))
    i_end   = inx_find(data_year_full, int(T_end)) + 1

    data_val  = data_val_full[:, i_start:i_end]
    data_year = data_year_full[i_start:i_end]

    return [data_val, data_year]


def check_error():

    [data_val, data_year] = cmip_temperature(scenerio_name='RCP2.6', T_start=1850, T_end=2023)


if __name__ == "__main__":
    check_error()
