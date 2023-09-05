import numpy as np
import netCDF4 as nc
import glob
from typing import List, Dict, Tuple, Union



def land_emission(folder_name='data/land_emission/'):

    file_name = folder_name + 'Gasser_et_al_2020_best_guess.nc'
    data = nc.Dataset(file_name)

    return data

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

def cmip_emission(scenerio_name: str, T_start: int = 1765, T_end: int = 2500, folder_name: str = 'data/emission/',emission_type:Union['fossil','fossil+land','land'] = 'fossil+land' ):

    assert T_start >= 1765
    assert T_end   <= 2500

    temp = np.loadtxt(folder_name + scenerio_name + '.txt')

    # 0. Year
    # 1. FossilCO2        - Fossil & Industrial CO2 (Fossil	Cement	Gas Flaring & Bunker Fuels)				
    # 2. OtherCO2         - Landuse related CO2 Emissions						
    # 3. CH4              - Methane						
    # 4. N2O              - Nitrous Oxide						
    # 5. - 11.            - Tropospheric ozone precursors	aerosols and reactive gas emissions					
    # 12. - 23.           - Flourinated gases controlled under the Kyoto Protocol	(HFCs	PFCs	SF6)			
    # 24. - 39.           - Ozone Depleting Substances controlled under the Montreal Protocol (CFCs	HFCFC	Halons	CCl4	MCF	CH3Br	CH3Cl)

    inx_time=0
    inx_co2_emission = [None]

    if emission_type=='fossil':
        inx_co2_emission=[1]
    if emission_type=='fossil+land':
        inx_co2_emission=[1,2]
    if emission_type=='land':
        inx_co2_emission=[2]

    data_val = np.array(np.sum(temp[:, inx_co2_emission], 1), dtype='float')
    data_year = np.array(temp[:, inx_time], dtype='int')

    # shortcut for finding index of given value
    inx_find = lambda x, val: int(np.where(x == val)[0])
    i_start = inx_find(data_year, int(T_start))
    i_end = inx_find(data_year, int(T_end)) + 1

    data_val = data_val[i_start:i_end]
    data_year = data_year[i_start:i_end]

    return [data_val, data_year]

def cmip_concentration(scenerio_name: str, T_start: int = 1765, T_end: int = 2500, folder_name: str = 'data/concentration/' ):

    assert T_start >= 1765
    assert T_end <= 2500

    temp = np.loadtxt(folder_name + scenerio_name + '.txt')

    #COLUMN_DESCRIPTION________________________________________
    #0. year
    #1. CO2EQ            - CO2 equivalence concentrations using CO2 radiative forcing relationship Q = 3.71/ln(2)*ln(C/278), aggregating all anthropogenic forcings, including greenhouse gases listed below (i.e. columns 3,4,5 and 8-35), and aerosols, trop. ozone etc. (not listed below).
    #2. KYOTO-CO2EQ      - As column 1, but only aggregating greenhouse gases controlled under the Kyoto Protocol (columns 3,4,5 and 8-19).
    #3. CO2              - Atmospheric CO2 concentrations
    #4. CH4              - Atmospheric CH4 concentrations
    #5. N2O              - Atmospheric N2O concentrations
    #6. FGASSUMHFC134AEQ - All flourinated gases controlled under the Kyoto Protocol, i.e. HFCs, PFCs, and SF6 (columns 8-19) expressed as HFC134a equivalence concentrations.
    #7. MHALOSUMCFC12EQ  - All flourinated gases controlled under the Montreal Protocol, i.e. CFCs, HCFCs, Halons, CCl4, CH3Br, CH3Cl (columns 20-35) expressed as CFC-12 equivalence concentrations.
    #8. - 19.            - Flourinated Gases controlled under the Kyoto Protocol
    #20. - 35.           - Ozone Depleting Substances controlled under the Montreal Protocol

    inx_time=0
    inx_co2_conc = 3

    data_val = np.array(temp[:, inx_co2_conc], dtype='float')
    data_year = np.array(temp[:, inx_time], dtype='int')

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
    val_set  = []
    t_min    = []

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
