CDF  �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 12:26:07 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp45/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp45/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-230012_globalmean_am_timeseries.nc
Sun Feb 28 12:26:04 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp45/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-230012_fulldata.nc /echam/folini/cmip5/historical_rcp45/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc
Sun Feb 28 12:25:58 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp45/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp45/full_data//IPSL-CM5A-MR_r1i1p1_ts_185001-230012_fulldata.nc
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
�   Aq��P   Aq�VP   C��iAq�{�   Aq�VP   Aq��P   C���Aq���   Aq��P   Aq�8P   C��Aq�]�   Aq�8P   Aq�P   C���Aq���   Aq�P   Aq�P   C���Aq�?�   Aq�P   Aq�P   C��Aq��   Aq�P   Aq��P   C��7Aq�!�   Aq��P   Aq�mP   C���Aq��   Aq�mP   Aq��P   C���Aq��   Aq��P   Aq�OP   C��;Aq�t�   Aq�OP   Aq��P   C���Aq���   Aq��P   Aq�1P   C���Aq�V�   Aq�1P   Aq��P   C��\Aq���   Aq��P   Aq�P   C�}�Aq�8�   Aq�P   Aq��P   C�n�Aq���   Aq��P   Aq��P   C�u"Aq��   Aq��P   ArfP   C���Ar��   ArfP   Ar�P   C��6Ar��   Ar�P   ArHP   C���Arm�   ArHP   Ar�P   C���Ar��   Ar�P   Ar*P   C��	ArO�   Ar*P   Ar�P   C��Ar��   Ar�P   ArP   C��@Ar1�   ArP   Ar}P   C��Ar��   Ar}P   Ar�P   C���Ar�   Ar�P   Ar_P   C���Ar��   Ar_P   Ar�P   C��bAr��   Ar�P   ArAP   C���Arf�   ArAP   Ar�P   C��=Ar��   Ar�P   Ar!#P   C��Ar!H�   Ar!#P   Ar#�P   C���Ar#��   Ar#�P   Ar&P   C���Ar&*�   Ar&P   Ar(vP   C��Ar(��   Ar(vP   Ar*�P   C���Ar+�   Ar*�P   Ar-XP   C���Ar-}�   Ar-XP   Ar/�P   C��jAr/��   Ar/�P   Ar2:P   C���Ar2_�   Ar2:P   Ar4�P   C��YAr4��   Ar4�P   Ar7P   C��+Ar7A�   Ar7P   Ar9�P   C��Ar9��   Ar9�P   Ar;�P   C��%Ar<#�   Ar;�P   Ar>oP   C���Ar>��   Ar>oP   Ar@�P   C���ArA�   Ar@�P   ArCQP   C��DArCv�   ArCQP   ArE�P   C��ArE��   ArE�P   ArH3P   C��ArHX�   ArH3P   ArJ�P   C��ArJ��   ArJ�P   ArMP   C���ArM:�   ArMP   ArO�P   C���ArO��   ArO�P   ArQ�P   C��FArR�   ArQ�P   ArThP   C��iArT��   ArThP   ArV�P   C��MArV��   ArV�P   ArYJP   C���ArYo�   ArYJP   Ar[�P   C���Ar[��   Ar[�P   Ar^,P   C���Ar^Q�   Ar^,P   Ar`�P   C��rAr`��   Ar`�P   ArcP   C���Arc3�   ArcP   AreP   C��Are��   AreP   Arg�P   C���Arh�   Arg�P   ArjaP   C���Arj��   ArjaP   Arl�P   C���Arl��   Arl�P   AroCP   C��XAroh�   AroCP   Arq�P   C���Arq��   Arq�P   Art%P   C���ArtJ�   Art%P   Arv�P   C��dArv��   Arv�P   AryP   C��\Ary,�   AryP   Ar{xP   C�ɁAr{��   Ar{xP   Ar}�P   C�ȴAr~�   Ar}�P   Ar�ZP   C��EAr��   Ar�ZP   Ar��P   C��Ar���   Ar��P   Ar�<P   C���Ar�a�   Ar�<P   Ar��P   C��kAr���   Ar��P   Ar�P   C�ĭAr�C�   Ar�P   Ar��P   C��.Ar���   Ar��P   Ar� P   C��LAr�%�   Ar� P   Ar�qP   C���Ar���   Ar�qP   Ar��P   C��2Ar��   Ar��P   Ar�SP   C��Ar�x�   Ar�SP   Ar��P   C��bAr���   Ar��P   Ar�5P   C�ڛAr�Z�   Ar�5P   Ar��P   C���Ar���   Ar��P   Ar�P   C��Ar�<�   Ar�P   Ar��P   C��}Ar���   Ar��P   Ar��P   C�ȗAr��   Ar��P   Ar�jP   C���Ar���   Ar�jP   Ar��P   C���Ar� �   Ar��P   Ar�LP   C�ȋAr�q�   Ar�LP   Ar��P   C�ҫAr���   Ar��P   Ar�.P   C��dAr�S�   Ar�.P   Ar��P   C���Ar���   Ar��P   Ar�P   C���Ar�5�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C���Ar��   Ar��P   Ar�cP   C��?Ar���   Ar�cP   Ar��P   C�͔Ar���   Ar��P   Ar�EP   C��3Ar�j�   Ar�EP   ArĶP   C���Ar���   ArĶP   Ar�'P   C��KAr�L�   Ar�'P   ArɘP   C�ÆArɽ�   ArɘP   Ar�	P   C��|Ar�.�   Ar�	P   Ar�zP   C���ArΟ�   Ar�zP   Ar��P   C���Ar��   Ar��P   Ar�\P   C��#ArӁ�   Ar�\P   Ar��P   C�ɊAr���   Ar��P   Ar�>P   C��gAr�c�   Ar�>P   ArگP   C�İAr���   ArگP   Ar� P   C�ܣAr�E�   Ar� P   ArߑP   C�׽Ar߶�   ArߑP   Ar�P   C��jAr�'�   Ar�P   Ar�sP   C��-Ar��   Ar�sP   Ar��P   C��Ar�	�   Ar��P   Ar�UP   C��Ar�z�   Ar�UP   Ar��P   C��Ar���   Ar��P   Ar�7P   C��?Ar�\�   Ar�7P   Ar�P   C��Ar���   Ar�P   Ar�P   C��Ar�>�   Ar�P   Ar��P   C��IAr���   Ar��P   Ar��P   C� �Ar� �   Ar��P   Ar�lP   C��Ar���   Ar�lP   Ar��P   C���Ar��   Ar��P   Ar�NP   C�ܕAr�s�   Ar�NP   As�P   C�� As��   As�P   As0P   C��jAsU�   As0P   As�P   C��6As��   As�P   As	P   C���As	7�   As	P   As�P   C��As��   As�P   As�P   C��As�   As�P   AseP   C��As��   AseP   As�P   C�~As��   As�P   AsGP   C� aAsl�   AsGP   As�P   C�OAs��   As�P   As)P   C�zAsN�   As)P   As�P   C�$-As��   As�P   AsP   C�#�As0�   AsP   As!|P   C�7As!��   As!|P   As#�P   C�.�As$�   As#�P   As&^P   C�$�As&��   As&^P   As(�P   C�8As(��   As(�P   As+@P   C�G�As+e�   As+@P   As-�P   C�G�As-��   As-�P   As0"P   C�J�As0G�   As0"P   As2�P   C�S�As2��   As2�P   As5P   C�O�As5)�   As5P   As7uP   C�OAs7��   As7uP   As9�P   C�SlAs:�   As9�P   As<WP   C�ZAs<|�   As<WP   As>�P   C�P�As>��   As>�P   AsA9P   C�RCAsA^�   AsA9P   AsC�P   C�_�AsC��   AsC�P   AsFP   C�a�AsF@�   AsFP   AsH�P   C�PbAsH��   AsH�P   AsJ�P   C�_�AsK"�   AsJ�P   AsMnP   C�y�AsM��   AsMnP   AsO�P   C�t�AsP�   AsO�P   AsRPP   C�xRAsRu�   AsRPP   AsT�P   C�x�AsT��   AsT�P   AsW2P   C�xAsWW�   AsW2P   AsY�P   C���AsY��   AsY�P   As\P   C���As\9�   As\P   As^�P   C���As^��   As^�P   As`�P   C��Asa�   As`�P   AscgP   C��Asc��   AscgP   Ase�P   C���Ase��   Ase�P   AshIP   C��5Ashn�   AshIP   Asj�P   C���Asj��   Asj�P   Asm+P   C���AsmP�   Asm+P   Aso�P   C���Aso��   Aso�P   AsrP   C���Asr2�   AsrP   Ast~P   C���Ast��   Ast~P   Asv�P   C�� Asw�   Asv�P   Asy`P   C��iAsy��   Asy`P   As{�P   C��As{��   As{�P   As~BP   C���As~g�   As~BP   As��P   C���As���   As��P   As�$P   C��cAs�I�   As�$P   As��P   C��PAs���   As��P   As�P   C���As�+�   As�P   As�wP   C��sAs���   As�wP   As��P   C���As��   As��P   As�YP   C��%As�~�   As�YP   As��P   C�ÓAs���   As��P   As�;P   C���As�`�   As�;P   As��P   C��As���   As��P   As�P   C��(As�B�   As�P   As��P   C���As���   As��P   As��P   C��VAs�$�   As��P   As�pP   C��&As���   As�pP   As��P   C���As��   As��P   As�RP   C��|As�w�   As�RP   As��P   C��;As���   As��P   As�4P   C���As�Y�   As�4P   As��P   C��jAs���   As��P   As�P   C���As�;�   As�P   As��P   C��uAs���   As��P   As��P   C� �As��   As��P   As�iP   C��As���   As�iP   As��P   C���As���   As��P   As�KP   C�As�p�   As�KP   As��P   C���As���   As��P   As�-P   C��As�R�   As�-P   AsP   C��As���   AsP   As�P   C��As�4�   As�P   AsǀP   C�$Asǥ�   AsǀP   As��P   C��HAs��   As��P   As�bP   C��Aṡ�   As�bP   As��P   C�?As���   As��P   As�DP   C��As�i�   As�DP   AsӵP   C��As���   AsӵP   As�&P   C�As�K�   As�&P   AsؗP   C�dAsؼ�   AsؗP   As�P   C��As�-�   As�P   As�yP   C��Asݞ�   As�yP   As��P   C�UAs��   As��P   As�[P   C�!As��   As�[P   As��P   C�As���   As��P   As�=P   C�|As�b�   As�=P   As�P   C��As���   As�P   As�P   C�,nAs�D�   As�P   As�P   C�As��   As�P   As�P   C��As�&�   As�P   As�rP   C�-As��   As�rP   As��P   C�*�As��   As��P   As�TP   C�YAs�y�   As�TP   As��P   C�?As���   As��P   As�6P   C�/As�[�   As�6P   As��P   C�2UAs���   As��P   AtP   C�$�At=�   AtP   At�P   C��At��   At�P   At�P   C�-�At�   At�P   At	kP   C�$�At	��   At	kP   At�P   C�1At�   At�P   AtMP   C�"�Atr�   AtMP   At�P   C�@kAt��   At�P   At/P   C�3�AtT�   At/P   At�P   C�'gAt��   At�P   AtP   C��At6�   AtP   At�P   C�&�At��   At�P   At�P   C�0*At�   At�P   AtdP   C�)�At��   AtdP   At!�P   C�%�At!��   At!�P   At$FP   C��At$k�   At$FP   At&�P   C�*�At&��   At&�P   At)(P   C�6�At)M�   At)(P   At+�P   C�&�At+��   At+�P   At.
P   C�'�At./�   At.
P   At0{P   C�;�At0��   At0{P   At2�P   C�eAt3�   At2�P   At5]P   C�,vAt5��   At5]P   At7�P   C�1�At7��   At7�P   At:?P   C�-oAt:d�   At:?P   At<�P   C�3^At<��   At<�P   At?!P   C�7?At?F�   At?!P   AtA�P   C�:*AtA��   AtA�P   AtDP   C�EZAtD(�   AtDP   AtFtP   C�BJAtF��   AtFtP   AtH�P   C�2$AtI
�   AtH�P   AtKVP   C�9hAtK{�   AtKVP   AtM�P   C�9?AtM��   AtM�P   AtP8P   C�<�AtP]�   AtP8P   AtR�P   C�5SAtR��   AtR�P   AtUP   C�0AtU?�   AtUP   AtW�P   C�B+AtW��   AtW�P   AtY�P   C�:AtZ!�   AtY�P   At\mP   C�9At\��   At\mP   At^�P   C�3At_�   At^�P   AtaOP   C�L[Atat�   AtaOP   Atc�P   C�WZAtc��   Atc�P   Atf1P   C�LIAtfV�   Atf1P   Ath�P   C�/RAth��   Ath�P   AtkP   C�:�Atk8�   AtkP   Atm�P   C�KAtm��   Atm�P   Ato�P   C�F�Atp�   Ato�P   AtrfP   C�4�Atr��   AtrfP   Att�P   C�3�Att��   Att�P   AtwHP   C�EfAtwm�   AtwHP   Aty�P   C�;fAty��   Aty�P   At|*P   C�<qAt|O�   At|*P   At~�P   C�I�At~��   At~�P   At�P   C�L�At�1�   At�P   At�}P   C�NVAt���   At�}P   At��P   C�?oAt��   At��P   At�_P   C�@_At���   At�_P   At��P   C�X	At���   At��P   At�AP   C�@rAt�f�   At�AP   At��P   C�X�At���   At��P   At�#P   C�S�At�H�   At�#P   At��P   C�NHAt���   At��P   At�P   C�F�At�*�   At�P   At�vP   C�CnAt���   At�vP   At��P   C�,�At��   At��P   At�XP   C�2�At�}�   At�XP   At��P   C�J9At���   At��P   At�:P   C�W?At�_�   At�:P   At��P   C�O�At���   At��P   At�P   C�P�At�A�   At�P   At��P   C�C:At���   At��P   At��P   C�IIAt�#�   At��P   At�oP   C�_�At���   At�oP   At��P   C�E�At��   At��P   At�QP   C�M�At�v�   At�QP   At��P   C�SAt���   At��P   At�3P   C�9MAt�X�   At�3P   At��P   C�F!At���   At��P   At�P   C�>�At�:�   At�P   At��P   C�H�At���   At��P   At��P   C�C{At��   At��P   At�hP   C�>�Atō�   At�hP   At��P   C�S@At���   At��P   At�JP   C�PxAt�o�   At�JP   At̻P   C�PAt���   At̻P   At�,P   C�J�At�Q�   At�,P   AtѝP   C�aKAt���   AtѝP   At�P   C�g;At�3�   At�P   At�P   C�_�At֤�   At�P   At��P   C�^vAt��   At��P   At�aP   C�Z,Atۆ�   At�aP   At��P   C�d1At���   At��P   At�CP   C�S�At�h�   At�CP   At�P   C�Q�At���   At�P   At�%P   C�Q�At�J�   At�%P   At�P   C�nAt��   At�P   At�P   C�Z�At�,�   At�P   At�xP   C�^�At��   At�xP   At��P   C�Z�At��   At��P   At�ZP   C�zwAt��   At�ZP   At��P   C�`�At���   At��P   At�<P   C�Q�At�a�   At�<P   At��P   C�M�At���   At��P   At�P   C�Q$At�C�   At�P   At��P   C�W�At���   At��P   Au  P   C�O�Au %�   Au  P   AuqP   C�TcAu��   AuqP   Au�P   C�B�Au�   Au�P   AuSP   C�A�Aux�   AuSP   Au	�P   C�JrAu	��   Au	�P   Au5P   C�GzAuZ�   Au5P   Au�P   C�J:Au��   Au�P   AuP   C�RYAu<�   AuP   Au�P   C�K�Au��   Au�P   Au�P   C�H�Au�   Au�P   AujP   C�[�Au��   AujP   Au�P   C�X�Au �   Au�P   AuLP   C�Q�Auq�   AuLP   Au�P   C�`NAu��   Au�P   Au".P   C�`/Au"S�   Au".P   Au$�P   C�Y�Au$��   Au$�P   Au'P   C�p�Au'5�   Au'P   Au)�P   C�f|Au)��   Au)�P   Au+�P   C�RAu,�   Au+�P   Au.cP   C�\!Au.��   Au.cP   Au0�P   C�TsAu0��   Au0�P   Au3EP   C�b�Au3j�   Au3EP   Au5�P   C�oKAu5��   Au5�P   Au8'P   C�g�Au8L�   Au8'P   Au:�P   C�_�Au:��   Au:�P   Au=	P   C�IuAu=.�   Au=	P   Au?zP   C�J�Au?��   Au?zP   AuA�P   C�a|AuB�   AuA�P   AuD\P   C�d[AuD��   AuD\P   AuF�P   C�G�AuF��   AuF�P   AuI>P   C�a�AuIc�   AuI>P   AuK�P   C�_xAuK��   AuK�P   AuN P   C�WAuNE�   AuN P   AuP�P   C�\�AuP��   AuP�P   AuSP   C�^�AuS'�   AuSP   AuUsP   C�T�AuU��   AuUsP   AuW�P   C�[�AuX	�   AuW�P   AuZUP   C�jmAuZz�   AuZUP   Au\�P   C�|HAu\��   Au\�P   Au_7P   C�]�Au_\�   Au_7P   Aua�P   C�V�Aua��   Aua�P   AudP   C�`�Aud>�   AudP   Auf�P   C�qOAuf��   Auf�P   Auh�P   C�{�Aui �   Auh�P   AuklP   C�o�Auk��   AuklP   Aum�P   C�i-Aun�   Aum�P   AupNP   C�t�Aups�   AupNP   Aur�P   C�h(Aur��   Aur�P   Auu0P   C�o�AuuU�   Auu0P   Auw�P   C�w�Auw��   Auw�P   AuzP   C�mmAuz7�   AuzP   Au|�P   C�kAu|��   Au|�P   Au~�P   C�~"Au�   Au~�P   Au�eP   C�`�Au���   Au�eP   Au��P   C�d	Au���   Au��P   Au�GP   C�oAu�l�   Au�GP   Au��P   C�j�Au���   Au��P   Au�)P   C�W�Au�N�   Au�)P   Au��P   C�~�Au���   Au��P   Au�P   C�}(Au�0�   Au�P   Au�|P   C�g�Au���   Au�|P   Au��P   C�eAu��   Au��P   Au�^P   C�Z`Au���   Au�^P   Au��P   C�g�Au���   Au��P   Au�@P   C�sEAu�e�   Au�@P   Au��P   C�t:Au���   Au��P   Au�"P   C�Au�G�   Au�"P   Au��P   C��>Au���   Au��P   Au�P   C�t Au�)�   Au�P   Au�uP   C�a�Au���   Au�uP   Au��P   C�o&Au��   Au��P   Au�WP   C�b\Au�|�   Au�WP   Au��P   C�wkAu���   Au��P   Au�9P   C�{Au�^�   Au�9P   Au��P   C�{�Au���   Au��P   Au�P   C�cEAu�@�   Au�P   Au��P   C�hVAu���   Au��P   Au��P   C�z�Au�"�   Au��P   Au�nP   C�vQAu���   Au�nP   Au��P   C�n�Au��   Au��P   Au�PP   C�qAu�u�   Au�PP   Au��P   C�i+Au���   Au��P   Au�2P   C��$Au�W�   Au�2P   AuʣP   C�~�Au���   AuʣP   Au�P   C�b�Au�9�   Au�P   AuυP   C�`AuϪ�   AuυP   Au��P   C�zrAu��   Au��P   Au�gP   C�opAuԌ�   Au�gP   Au��P   C��4Au���   Au��P   Au�IP   C���Au�n�   Au�IP   AuۺP   C�_vAu���   AuۺP   Au�+P   C�^�Au�P�   Au�+P   Au��P   C�oAu���   Au��P   Au�P   C�f�Au�2�   Au�P   Au�~P   C�_�Au��   Au�~P   Au��P   C�kAu��   Au��P   Au�`P   C�o�Au��   Au�`P   Au��P   C�j�Au���   Au��P   Au�BP   C�wAu�g�   Au�BP   Au�P   C���