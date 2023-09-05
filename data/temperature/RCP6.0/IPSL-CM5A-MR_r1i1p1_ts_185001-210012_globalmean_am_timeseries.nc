CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 12:00:08 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp60/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp60/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-210012_globalmean_am_timeseries.nc
Sun Feb 28 12:00:06 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp60/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-210012_fulldata.nc /echam/folini/cmip5/historical_rcp60/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc
Sun Feb 28 12:00:02 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp60/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp60/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-210012_fulldata.nc
Sat May 06 11:37:05 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:37:01 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/IPSL-CM5A-MR/r1i1p1/ts_Amon_IPSL-CM5A-MR_historical_r1i1p1_185001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
2011-09-22T19:36:32Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �IPSL-CM5A-MR (2010) : atmos : LMDZ4 (LMDZ4_v5, 144x143x39); ocean : ORCA2 (NEMOV2_3, 2x2L31); seaIce : LIM2 (NEMOV2_3); ocnBgchem : PISCES (NEMOV2_3); land : ORCHIDEE (orchidee_1_9_4_AR5)    institution       3IPSL (Institut Pierre Simon Laplace, Paris, France)    institute_id      IPSL   experiment_id         
historical     model_id      IPSL-CM5A-MR   forcing       &Nat,Ant,GHG,SA,Oz,LU,SS,Ds,BC,MD,OC,AA     parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @��        contact       ?ipsl-cmip5 _at_ ipsl.jussieu.fr Data manager : Sebastien Denvil    comment       HThis 20th century simulation include natural and anthropogenic forcings.   
references        NModel documentation and further reference available here : http://icmc.ipsl.fr     initialization_method               physics_version             tracking_id       $cd26d4ae-8882-4c11-ab3f-a7af4394c2ea   product       output     
experiment        
historical     	frequency         year   creation_date         2011-09-22T19:36:32Z   
project_id        CMIP5      table_id      =Table Amon (31 January 2011) 53b766a395ac41696af40aab76a49ae5      title         7IPSL-CM5A-MR model output prepared for CMIP5 historical    parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.7.1      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               	time_bnds                             (   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X              lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y              ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         tsol       cell_methods      "time: mean (interval: 30 minutes)      history       �2011-09-22T19:36:31Z altered by CMOR: replaced missing value flag (9.96921e+36) with standard missing value (1e+20). 2011-09-22T19:36:32Z altered by CMOR: Inverted axis: lat.     associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_IPSL-CM5A-MR_historical_r0i0p0.nc areacella: areacella_fx_IPSL-CM5A-MR_historical_r0i0p0.nc          8                Aq���   Aq��P   Aq�P   C��cAq�6�   Aq�P   Aq��P   C��JAq���   Aq��P   Aq��P   C��HAq��   Aq��P   Aq�dP   C��Aq���   Aq�dP   Aq��P   C��[Aq���   Aq��P   Aq�FP   C���Aq�k�   Aq�FP   Aq��P   C���Aq���   Aq��P   Aq�(P   C��Aq�M�   Aq�(P   Aq��P   C��|Aq���   Aq��P   Aq�
P   C���Aq�/�   Aq�
P   Aq�{P   C���Aq���   Aq�{P   Aq��P   C���Aq��   Aq��P   Aq�]P   C���AqĂ�   Aq�]P   Aq��P   C���Aq���   Aq��P   Aq�?P   C��mAq�d�   Aq�?P   Aq˰P   C���Aq���   Aq˰P   Aq�!P   C��*Aq�F�   Aq�!P   AqВP   C���Aqз�   AqВP   Aq�P   C���Aq�(�   Aq�P   Aq�tP   C���Aqՙ�   Aq�tP   Aq��P   C���Aq�
�   Aq��P   Aq�VP   C��iAq�{�   Aq�VP   Aq��P   C���Aq���   Aq��P   Aq�8P   C��Aq�]�   Aq�8P   Aq�P   C���Aq���   Aq�P   Aq�P   C���Aq�?�   Aq�P   Aq�P   C��Aq��   Aq�P   Aq��P   C��7Aq�!�   Aq��P   Aq�mP   C���Aq��   Aq�mP   Aq��P   C���Aq��   Aq��P   Aq�OP   C��;Aq�t�   Aq�OP   Aq��P   C���Aq���   Aq��P   Aq�1P   C���Aq�V�   Aq�1P   Aq��P   C��\Aq���   Aq��P   Aq�P   C�}�Aq�8�   Aq�P   Aq��P   C�n�Aq���   Aq��P   Aq��P   C�u"Aq��   Aq��P   ArfP   C���Ar��   ArfP   Ar�P   C��6Ar��   Ar�P   ArHP   C���Arm�   ArHP   Ar�P   C���Ar��   Ar�P   Ar*P   C��	ArO�   Ar*P   Ar�P   C��Ar��   Ar�P   ArP   C��@Ar1�   ArP   Ar}P   C��Ar��   Ar}P   Ar�P   C���Ar�   Ar�P   Ar_P   C���Ar��   Ar_P   Ar�P   C��bAr��   Ar�P   ArAP   C���Arf�   ArAP   Ar�P   C��=Ar��   Ar�P   Ar!#P   C��Ar!H�   Ar!#P   Ar#�P   C���Ar#��   Ar#�P   Ar&P   C���Ar&*�   Ar&P   Ar(vP   C��Ar(��   Ar(vP   Ar*�P   C���Ar+�   Ar*�P   Ar-XP   C���Ar-}�   Ar-XP   Ar/�P   C��jAr/��   Ar/�P   Ar2:P   C���Ar2_�   Ar2:P   Ar4�P   C��YAr4��   Ar4�P   Ar7P   C��+Ar7A�   Ar7P   Ar9�P   C��Ar9��   Ar9�P   Ar;�P   C��%Ar<#�   Ar;�P   Ar>oP   C���Ar>��   Ar>oP   Ar@�P   C���ArA�   Ar@�P   ArCQP   C��DArCv�   ArCQP   ArE�P   C��ArE��   ArE�P   ArH3P   C��ArHX�   ArH3P   ArJ�P   C��ArJ��   ArJ�P   ArMP   C���ArM:�   ArMP   ArO�P   C���ArO��   ArO�P   ArQ�P   C��FArR�   ArQ�P   ArThP   C��iArT��   ArThP   ArV�P   C��MArV��   ArV�P   ArYJP   C���ArYo�   ArYJP   Ar[�P   C���Ar[��   Ar[�P   Ar^,P   C���Ar^Q�   Ar^,P   Ar`�P   C��rAr`��   Ar`�P   ArcP   C���Arc3�   ArcP   AreP   C��Are��   AreP   Arg�P   C���Arh�   Arg�P   ArjaP   C���Arj��   ArjaP   Arl�P   C���Arl��   Arl�P   AroCP   C��XAroh�   AroCP   Arq�P   C���Arq��   Arq�P   Art%P   C���ArtJ�   Art%P   Arv�P   C��dArv��   Arv�P   AryP   C��\Ary,�   AryP   Ar{xP   C�ɁAr{��   Ar{xP   Ar}�P   C�ȴAr~�   Ar}�P   Ar�ZP   C��EAr��   Ar�ZP   Ar��P   C��Ar���   Ar��P   Ar�<P   C���Ar�a�   Ar�<P   Ar��P   C��kAr���   Ar��P   Ar�P   C�ĭAr�C�   Ar�P   Ar��P   C��.Ar���   Ar��P   Ar� P   C��LAr�%�   Ar� P   Ar�qP   C���Ar���   Ar�qP   Ar��P   C��2Ar��   Ar��P   Ar�SP   C��Ar�x�   Ar�SP   Ar��P   C��bAr���   Ar��P   Ar�5P   C�ڛAr�Z�   Ar�5P   Ar��P   C���Ar���   Ar��P   Ar�P   C��Ar�<�   Ar�P   Ar��P   C��}Ar���   Ar��P   Ar��P   C�ȗAr��   Ar��P   Ar�jP   C���Ar���   Ar�jP   Ar��P   C���Ar� �   Ar��P   Ar�LP   C�ȋAr�q�   Ar�LP   Ar��P   C�ҫAr���   Ar��P   Ar�.P   C��dAr�S�   Ar�.P   Ar��P   C���Ar���   Ar��P   Ar�P   C���Ar�5�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C���Ar��   Ar��P   Ar�cP   C��?Ar���   Ar�cP   Ar��P   C�͔Ar���   Ar��P   Ar�EP   C��3Ar�j�   Ar�EP   ArĶP   C���Ar���   ArĶP   Ar�'P   C��KAr�L�   Ar�'P   ArɘP   C�ÆArɽ�   ArɘP   Ar�	P   C��|Ar�.�   Ar�	P   Ar�zP   C���ArΟ�   Ar�zP   Ar��P   C���Ar��   Ar��P   Ar�\P   C��#ArӁ�   Ar�\P   Ar��P   C�ɊAr���   Ar��P   Ar�>P   C��gAr�c�   Ar�>P   ArگP   C�İAr���   ArگP   Ar� P   C�ܣAr�E�   Ar� P   ArߑP   C�׽Ar߶�   ArߑP   Ar�P   C��jAr�'�   Ar�P   Ar�sP   C��-Ar��   Ar�sP   Ar��P   C��Ar�	�   Ar��P   Ar�UP   C��Ar�z�   Ar�UP   Ar��P   C��Ar���   Ar��P   Ar�7P   C��?Ar�\�   Ar�7P   Ar�P   C��Ar���   Ar�P   Ar�P   C��Ar�>�   Ar�P   Ar��P   C��IAr���   Ar��P   Ar��P   C� �Ar� �   Ar��P   Ar�lP   C��Ar���   Ar�lP   Ar��P   C���Ar��   Ar��P   Ar�NP   C�ܕAr�s�   Ar�NP   As�P   C�� As��   As�P   As0P   C��jAsU�   As0P   As�P   C��6As��   As�P   As	P   C���As	7�   As	P   As�P   C��As��   As�P   As�P   C��As�   As�P   AseP   C��As��   AseP   As�P   C�~As��   As�P   AsGP   C� aAsl�   AsGP   As�P   C�OAs��   As�P   As)P   C�zAsN�   As)P   As�P   C�$-As��   As�P   AsP   C�#�As0�   AsP   As!|P   C�7As!��   As!|P   As#�P   C�/�As$�   As#�P   As&^P   C�"�As&��   As&^P   As(�P   C�-HAs(��   As(�P   As+@P   C�AAs+e�   As+@P   As-�P   C�;�As-��   As-�P   As0"P   C�Q�As0G�   As0"P   As2�P   C�N�As2��   As2�P   As5P   C�P�As5)�   As5P   As7uP   C�U�As7��   As7uP   As9�P   C�`QAs:�   As9�P   As<WP   C�Q<As<|�   As<WP   As>�P   C�H�As>��   As>�P   AsA9P   C�R�AsA^�   AsA9P   AsC�P   C�Y�AsC��   AsC�P   AsFP   C�XKAsF@�   AsFP   AsH�P   C�_cAsH��   AsH�P   AsJ�P   C�kIAsK"�   AsJ�P   AsMnP   C�a<AsM��   AsMnP   AsO�P   C�MAsP�   AsO�P   AsRPP   C�Q=AsRu�   AsRPP   AsT�P   C�bnAsT��   AsT�P   AsW2P   C�l�AsWW�   AsW2P   AsY�P   C�|.AsY��   AsY�P   As\P   C��JAs\9�   As\P   As^�P   C�j)As^��   As^�P   As`�P   C�{TAsa�   As`�P   AscgP   C�lcAsc��   AscgP   Ase�P   C���Ase��   Ase�P   AshIP   C�� Ashn�   AshIP   Asj�P   C���Asj��   Asj�P   Asm+P   C�z�AsmP�   Asm+P   Aso�P   C��Aso��   Aso�P   AsrP   C��Asr2�   AsrP   Ast~P   C��TAst��   Ast~P   Asv�P   C��CAsw�   Asv�P   Asy`P   C��_Asy��   Asy`P   As{�P   C��As{��   As{�P   As~BP   C���As~g�   As~BP   As��P   C��1As���   As��P   As�$P   C���As�I�   As�$P   As��P   C��&As���   As��P   As�P   C��:As�+�   As�P   As�wP   C��SAs���   As�wP   As��P   C��'As��   As��P   As�YP   C���As�~�   As�YP   As��P   C��As���   As��P   As�;P   C���As�`�   As�;P   As��P   C���As���   As��P   As�P   C���As�B�   As�P   As��P   C�ܠAs���   As��P   As��P   C��As�$�   As��P   As�pP   C���As���   As�pP   As��P   C��As��   As��P   As�RP   C��As�w�   As�RP   As��P   C��3As���   As��P   As�4P   C���As�Y�   As�4P   As��P   C�ۧAs���   As��P   As�P   C���As�;�   As�P   As��P   C�ہAs���   As��P   As��P   C��As��   As��P   As�iP   C��As���   As�iP   As��P   C��GAs���   As��P   As�KP   C���As�p�   As�KP   As��P   C���As���   As��P   As�-P   C��aAs�R�   As�-P   AsP   C��As���   AsP   As�P   C��;As�4�   As�P   AsǀP   C�
�Asǥ�   AsǀP   As��P   C��As��   As��P   As�bP   C� Aṡ�   As�bP   As��P   C�&�As���   As��P   As�DP   C�%�As�i�   As�DP   AsӵP   C��As���   AsӵP   As�&P   C�,IAs�K�   As�&P   AsؗP   C�&�Asؼ�   AsؗP   As�P   C�/�As�-�   As�P   As�yP   C�C�Asݞ�   As�yP   As��P   C�0As��   As��P   As�[P   C�:1As��   As�[P   As��P   C�C�As���   As��P   As�=P   C�W�As�b�   As�=P   As�P   C�O�As���   As�P   As�P   C�P�As�D�   As�P   As�P   C�a�As��   As�P   As�P   C�b�As�&�   As�P   As�rP   C�Z�As��   As�rP   As��P   C�B�As��   As��P   As�TP   C�X&As�y�   As�TP   As��P   C�\bAs���   As��P   As�6P   C�wAs�[�   As�6P   As��P   C�o,As���   As��P   AtP   C�X�At=�   AtP   At�P   C�K�At��   At�P   At�P   C�\LAt�   At�P   At	kP   C�ku