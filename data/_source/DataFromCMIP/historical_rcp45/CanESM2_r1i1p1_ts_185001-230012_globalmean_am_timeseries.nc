CDF  �   
      time       bnds      lon       lat          !   CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 12:12:47 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp45/full_data//CanESM2_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp45/full_data//CanESM2_r1i1p1_ts_185001-230012_globalmean_am_timeseries.nc
Sun Feb 28 12:12:45 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp45/full_data//CanESM2_r1i1p1_ts_185001-230012_fulldata.nc /echam/folini/cmip5/historical_rcp45/full_data//CanESM2_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc
Sun Feb 28 12:12:43 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//CanESM2_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp45/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp45/full_data//CanESM2_r1i1p1_ts_185001-230012_fulldata.nc
Sat May 06 11:34:06 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:34:04 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/CanESM2/r1i1p1/ts_Amon_CanESM2_historical_r1i1p1_185001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
2011-03-16T18:50:42Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �CanESM2 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) and CMOC1.2 sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7 and CTEM1      institution       PCCCma (Canadian Centre for Climate Modelling and Analysis, Victoria, BC, Canada)   institute_id      CCCma      experiment_id         
historical     model_id      CanESM2    forcing       IGHG,Oz,SA,BC,OC,LU,Sl,Vl (GHG includes CO2,CH4,N2O,CFC11,effective CFC12)      parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       A�X       contact       cccma_info@ec.gc.ca    
references         http://www.cccma.ec.gc.ca/models   initialization_method               physics_version             tracking_id       $1490335b-daaf-42dc-b307-759309980803   branch_time_YMDH      2321:01:01:00      CCCma_runid       IGM    CCCma_parent_runid        IGA    CCCma_data_licence       �1) GRANT OF LICENCE - The Government of Canada (Environment Canada) is the 
owner of all intellectual property rights (including copyright) that may exist in this Data 
product. You (as "The Licensee") are hereby granted a non-exclusive, non-assignable, 
non-transferable unrestricted licence to use this data product for any purpose including 
the right to share these data with others and to make value-added and derivative 
products from it. This licence is not a sale of any or all of the owner's rights.
2) NO WARRANTY - This Data product is provided "as-is"; it has not been designed or 
prepared to meet the Licensee's particular requirements. Environment Canada makes no 
warranty, either express or implied, including but not limited to, warranties of 
merchantability and fitness for a particular purpose. In no event will Environment Canada 
be liable for any indirect, special, consequential or other damages attributed to the 
Licensee's use of the Data product.    product       output     
experiment        
historical     	frequency         year   creation_date         2011-03-16T18:50:42Z   
project_id        CMIP5      table_id      =Table Amon (31 January 2011) 53b766a395ac41696af40aab76a49ae5      title         2CanESM2 model output prepared for CMIP5 historical     parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.5.4      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T           �   	time_bnds                             �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           t   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           |   ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         GT     cell_methods      "time: mean (interval: 15 minutes)      history       o2011-03-16T18:50:42Z altered by CMOR: replaced missing value flag (1e+38) with standard missing value (1e+20).     associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CanESM2_historical_r0i0p0.nc areacella: areacella_fx_CanESM2_historical_r0i0p0.nc            �                Aq���   Aq��P   Aq�P   C�s�Aq�6�   Aq�P   Aq��P   C��Aq���   Aq��P   Aq��P   C��Aq��   Aq��P   Aq�dP   C��OAq���   Aq�dP   Aq��P   C���Aq���   Aq��P   Aq�FP   C��gAq�k�   Aq�FP   Aq��P   C���Aq���   Aq��P   Aq�(P   C�nPAq�M�   Aq�(P   Aq��P   C�yoAq���   Aq��P   Aq�
P   C�e�Aq�/�   Aq�
P   Aq�{P   C�|XAq���   Aq�{P   Aq��P   C���Aq��   Aq��P   Aq�]P   C�~�AqĂ�   Aq�]P   Aq��P   C�k�Aq���   Aq��P   Aq�?P   C�j�Aq�d�   Aq�?P   Aq˰P   C�mAq���   Aq˰P   Aq�!P   C�i(Aq�F�   Aq�!P   AqВP   C���Aqз�   AqВP   Aq�P   C��Aq�(�   Aq�P   Aq�tP   C�w?Aqՙ�   Aq�tP   Aq��P   C���Aq�
�   Aq��P   Aq�VP   C���Aq�{�   Aq�VP   Aq��P   C���Aq���   Aq��P   Aq�8P   C��zAq�]�   Aq�8P   Aq�P   C���Aq���   Aq�P   Aq�P   C���Aq�?�   Aq�P   Aq�P   C��Aq��   Aq�P   Aq��P   C���Aq�!�   Aq��P   Aq�mP   C���Aq��   Aq�mP   Aq��P   C��kAq��   Aq��P   Aq�OP   C���Aq�t�   Aq�OP   Aq��P   C��hAq���   Aq��P   Aq�1P   C��TAq�V�   Aq�1P   Aq��P   C�w�Aq���   Aq��P   Aq�P   C�^�Aq�8�   Aq�P   Aq��P   C�P�Aq���   Aq��P   Aq��P   C�F�Aq��   Aq��P   ArfP   C�_-Ar��   ArfP   Ar�P   C�XXAr��   Ar�P   ArHP   C�^�Arm�   ArHP   Ar�P   C�\�Ar��   Ar�P   Ar*P   C�l1ArO�   Ar*P   Ar�P   C��Ar��   Ar�P   ArP   C���Ar1�   ArP   Ar}P   C�{dAr��   Ar}P   Ar�P   C���Ar�   Ar�P   Ar_P   C���Ar��   Ar_P   Ar�P   C��NAr��   Ar�P   ArAP   C�n!Arf�   ArAP   Ar�P   C��/Ar��   Ar�P   Ar!#P   C���Ar!H�   Ar!#P   Ar#�P   C�{nAr#��   Ar#�P   Ar&P   C�hAr&*�   Ar&P   Ar(vP   C�q�Ar(��   Ar(vP   Ar*�P   C�k�Ar+�   Ar*�P   Ar-XP   C�c8Ar-}�   Ar-XP   Ar/�P   C��Ar/��   Ar/�P   Ar2:P   C���Ar2_�   Ar2:P   Ar4�P   C���Ar4��   Ar4�P   Ar7P   C�u�Ar7A�   Ar7P   Ar9�P   C�q&Ar9��   Ar9�P   Ar;�P   C���Ar<#�   Ar;�P   Ar>oP   C���Ar>��   Ar>oP   Ar@�P   C���ArA�   Ar@�P   ArCQP   C�~]ArCv�   ArCQP   ArE�P   C��8ArE��   ArE�P   ArH3P   C��)ArHX�   ArH3P   ArJ�P   C��ArJ��   ArJ�P   ArMP   C��ArM:�   ArMP   ArO�P   C��ArO��   ArO�P   ArQ�P   C��(ArR�   ArQ�P   ArThP   C��RArT��   ArThP   ArV�P   C���ArV��   ArV�P   ArYJP   C��ArYo�   ArYJP   Ar[�P   C���Ar[��   Ar[�P   Ar^,P   C��Ar^Q�   Ar^,P   Ar`�P   C��NAr`��   Ar`�P   ArcP   C��cArc3�   ArcP   AreP   C���Are��   AreP   Arg�P   C��Arh�   Arg�P   ArjaP   C��YArj��   ArjaP   Arl�P   C���Arl��   Arl�P   AroCP   C���Aroh�   AroCP   Arq�P   C���Arq��   Arq�P   Art%P   C��lArtJ�   Art%P   Arv�P   C���Arv��   Arv�P   AryP   C���Ary,�   AryP   Ar{xP   C���Ar{��   Ar{xP   Ar}�P   C��rAr~�   Ar}�P   Ar�ZP   C���Ar��   Ar�ZP   Ar��P   C��/Ar���   Ar��P   Ar�<P   C��*Ar�a�   Ar�<P   Ar��P   C���Ar���   Ar��P   Ar�P   C��~Ar�C�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar� P   C���Ar�%�   Ar� P   Ar�qP   C�~�Ar���   Ar�qP   Ar��P   C���Ar��   Ar��P   Ar�SP   C��9Ar�x�   Ar�SP   Ar��P   C��lAr���   Ar��P   Ar�5P   C��HAr�Z�   Ar�5P   Ar��P   C���Ar���   Ar��P   Ar�P   C��;Ar�<�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar��P   C���Ar��   Ar��P   Ar�jP   C��6Ar���   Ar�jP   Ar��P   C��;Ar� �   Ar��P   Ar�LP   C��7Ar�q�   Ar�LP   Ar��P   C��:Ar���   Ar��P   Ar�.P   C���Ar�S�   Ar�.P   Ar��P   C���Ar���   Ar��P   Ar�P   C��bAr�5�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C��0Ar��   Ar��P   Ar�cP   C��Ar���   Ar�cP   Ar��P   C��mAr���   Ar��P   Ar�EP   C���Ar�j�   Ar�EP   ArĶP   C�|PAr���   ArĶP   Ar�'P   C���Ar�L�   Ar�'P   ArɘP   C��PArɽ�   ArɘP   Ar�	P   C���Ar�.�   Ar�	P   Ar�zP   C���ArΟ�   Ar�zP   Ar��P   C��&Ar��   Ar��P   Ar�\P   C��FArӁ�   Ar�\P   Ar��P   C���Ar���   Ar��P   Ar�>P   C��!Ar�c�   Ar�>P   ArگP   C���Ar���   ArگP   Ar� P   C���Ar�E�   Ar� P   ArߑP   C��2Ar߶�   ArߑP   Ar�P   C���Ar�'�   Ar�P   Ar�sP   C��xAr��   Ar�sP   Ar��P   C���Ar�	�   Ar��P   Ar�UP   C�ЪAr�z�   Ar�UP   Ar��P   C��Ar���   Ar��P   Ar�7P   C��XAr�\�   Ar�7P   Ar�P   C��|Ar���   Ar�P   Ar�P   C��\Ar�>�   Ar�P   Ar��P   C��vAr���   Ar��P   Ar��P   C���Ar� �   Ar��P   Ar�lP   C�־Ar���   Ar�lP   Ar��P   C��-Ar��   Ar��P   Ar�NP   C�ӳAr�s�   Ar�NP   As�P   C��0As��   As�P   As0P   C��PAsU�   As0P   As�P   C��KAs��   As�P   As	P   C��As	7�   As	P   As�P   C���As��   As�P   As�P   C�FAs�   As�P   AseP   C��As��   AseP   As�P   C��As��   As�P   AsGP   C�� Asl�   AsGP   As�P   C�� As��   As�P   As)P   C��AsN�   As)P   As�P   C�SAs��   As�P   AsP   C�As0�   AsP   As!|P   C�0jAs!��   As!|P   As#�P   C�?-As$�   As#�P   As&^P   C��As&��   As&^P   As(�P   C�0As(��   As(�P   As+@P   C� �As+e�   As+@P   As-�P   C�-As-��   As-�P   As0"P   C�-�As0G�   As0"P   As2�P   C�*As2��   As2�P   As5P   C��As5)�   As5P   As7uP   C�1zAs7��   As7uP   As9�P   C�7�As:�   As9�P   As<WP   C�>zAs<|�   As<WP   As>�P   C�=BAs>��   As>�P   AsA9P   C�3�AsA^�   AsA9P   AsC�P   C�7�AsC��   AsC�P   AsFP   C�KnAsF@�   AsFP   AsH�P   C�PSAsH��   AsH�P   AsJ�P   C�[�AsK"�   AsJ�P   AsMnP   C�ZRAsM��   AsMnP   AsO�P   C�[<AsP�   AsO�P   AsRPP   C�b�AsRu�   AsRPP   AsT�P   C�n�AsT��   AsT�P   AsW2P   C�s�AsWW�   AsW2P   AsY�P   C�d	AsY��   AsY�P   As\P   C�}\As\9�   As\P   As^�P   C��_As^��   As^�P   As`�P   C�~�Asa�   As`�P   AscgP   C�oCAsc��   AscgP   Ase�P   C��5Ase��   Ase�P   AshIP   C�~VAshn�   AshIP   Asj�P   C���Asj��   Asj�P   Asm+P   C��hAsmP�   Asm+P   Aso�P   C���Aso��   Aso�P   AsrP   C���Asr2�   AsrP   Ast~P   C��1Ast��   Ast~P   Asv�P   C���Asw�   Asv�P   Asy`P   C���Asy��   Asy`P   As{�P   C��-As{��   As{�P   As~BP   C��-As~g�   As~BP   As��P   C���As���   As��P   As�$P   C���As�I�   As�$P   As��P   C��As���   As��P   As�P   C���As�+�   As�P   As�wP   C��As���   As�wP   As��P   C��~As��   As��P   As�YP   C���As�~�   As�YP   As��P   C��^As���   As��P   As�;P   C��As�`�   As�;P   As��P   C���As���   As��P   As�P   C���As�B�   As�P   As��P   C��2As���   As��P   As��P   C��!As�$�   As��P   As�pP   C�ҭAs���   As�pP   As��P   C��As��   As��P   As�RP   C��xAs�w�   As�RP   As��P   C���As���   As��P   As�4P   C��^As�Y�   As�4P   As��P   C��@As���   As��P   As�P   C��7As�;�   As�P   As��P   C��-As���   As��P   As��P   C�ظAs��   As��P   As�iP   C��<As���   As�iP   As��P   C��As���   As��P   As�KP   C��As�p�   As�KP   As��P   C��pAs���   As��P   As�-P   C�� As�R�   As�-P   AsP   C��lAs���   AsP   As�P   C���As�4�   As�P   AsǀP   C��9Asǥ�   AsǀP   As��P   C���As��   As��P   As�bP   C���Aṡ�   As�bP   As��P   C��;As���   As��P   As�DP   C��As�i�   As�DP   AsӵP   C�;As���   AsӵP   As�&P   C��As�K�   As�&P   AsؗP   C�(bAsؼ�   AsؗP   As�P   C� gAs�-�   As�P   As�yP   C�jAsݞ�   As�yP   As��P   C�^As��   As��P   As�[P   C��As��   As�[P   As��P   C�SAs���   As��P   As�=P   C�%�As�b�   As�=P   As�P   C��As���   As�P   As�P   C�As�D�   As�P   As�P   C�3As��   As�P   As�P   C���As�&�   As�P   As�rP   C��As��   As�rP   As��P   C�� As��   As��P   As�TP   C�
(As�y�   As�TP   As��P   C�@As���   As��P   As�6P   C��As�[�   As�6P   As��P   C��As���   As��P   AtP   C�At=�   AtP   At�P   C�'�At��   At�P   At�P   C�"TAt�   At�P   At	kP   C��At	��   At	kP   At�P   C�!�At�   At�P   AtMP   C��Atr�   AtMP   At�P   C�At��   At�P   At/P   C�*AtT�   At/P   At�P   C�"At��   At�P   AtP   C�At6�   AtP   At�P   C� �At��   At�P   At�P   C� �At�   At�P   AtdP   C�,-At��   AtdP   At!�P   C�)GAt!��   At!�P   At$FP   C�dAt$k�   At$FP   At&�P   C�3At&��   At&�P   At)(P   C�CPAt)M�   At)(P   At+�P   C�I�At+��   At+�P   At.
P   C�25At./�   At.
P   At0{P   C��At0��   At0{P   At2�P   C��At3�   At2�P   At5]P   C��At5��   At5]P   At7�P   C�3�At7��   At7�P   At:?P   C�, At:d�   At:?P   At<�P   C�!�At<��   At<�P   At?!P   C�7,At?F�   At?!P   AtA�P   C�6hAtA��   AtA�P   AtDP   C��AtD(�   AtDP   AtFtP   C�|AtF��   AtFtP   AtH�P   C�2IAtI
�   AtH�P   AtKVP   C�3�AtK{�   AtKVP   AtM�P   C�>CAtM��   AtM�P   AtP8P   C�)�AtP]�   AtP8P   AtR�P   C�/UAtR��   AtR�P   AtUP   C�4�AtU?�   AtUP   AtW�P   C�ISAtW��   AtW�P   AtY�P   C�-�AtZ!�   AtY�P   At\mP   C�.At\��   At\mP   At^�P   C�3�At_�   At^�P   AtaOP   C�-�Atat�   AtaOP   Atc�P   C�.*Atc��   Atc�P   Atf1P   C�CtAtfV�   Atf1P   Ath�P   C�H�Ath��   Ath�P   AtkP   C�>�Atk8�   AtkP   Atm�P   C�2�Atm��   Atm�P   Ato�P   C�.�Atp�   Ato�P   AtrfP   C�6�Atr��   AtrfP   Att�P   C�@BAtt��   Att�P   AtwHP   C�.�Atwm�   AtwHP   Aty�P   C�'�Aty��   Aty�P   At|*P   C�%|At|O�   At|*P   At~�P   C�8At~��   At~�P   At�P   C�D6At�1�   At�P   At�}P   C�;At���   At�}P   At��P   C�9At��   At��P   At�_P   C�D�At���   At�_P   At��P   C�1At���   At��P   At�AP   C�7�At�f�   At�AP   At��P   C�;XAt���   At��P   At�#P   C�E�At�H�   At�#P   At��P   C�C�At���   At��P   At�P   C�G
At�*�   At�P   At�vP   C�K�At���   At�vP   At��P   C�K�At��   At��P   At�XP   C�Y�At�}�   At�XP   At��P   C�@�At���   At��P   At�:P   C�3�At�_�   At�:P   At��P   C�%�At���   At��P   At�P   C�*,At�A�   At�P   At��P   C�D�At���   At��P   At��P   C�S\At�#�   At��P   At�oP   C�C�At���   At�oP   At��P   C�+>At��   At��P   At�QP   C�%At�v�   At�QP   At��P   C�F�At���   At��P   At�3P   C�YKAt�X�   At�3P   At��P   C�>%At���   At��P   At�P   C�#2At�:�   At�P   At��P   C��At���   At��P   At��P   C�<?At��   At��P   At�hP   C�=�Atō�   At�hP   At��P   C�@|At���   At��P   At�JP   C�<WAt�o�   At�JP   At̻P   C�0�At���   At̻P   At�,P   C�2UAt�Q�   At�,P   AtѝP   C�ApAt���   AtѝP   At�P   C�K�At�3�   At�P   At�P   C�5�At֤�   At�P   At��P   C�6<At��   At��P   At�aP   C�0*Atۆ�   At�aP   At��P   C�9�At���   At��P   At�CP   C�YpAt�h�   At�CP   At�P   C�YjAt���   At�P   At�%P   C�);At�J�   At�%P   At�P   C�&�At��   At�P   At�P   C�="At�,�   At�P   At�xP   C�DAt��   At�xP   At��P   C�X!At��   At��P   At�ZP   C�E2At��   At�ZP   At��P   C�J�At���   At��P   At�<P   C�U~At�a�   At�<P   At��P   C�W�At���   At��P   At�P   C�H�At�C�   At�P   At��P   C�E^At���   At��P   Au  P   C�*YAu %�   Au  P   AuqP   C�(BAu��   AuqP   Au�P   C�>Au�   Au�P   AuSP   C�[Aux�   AuSP   Au	�P   C�=VAu	��   Au	�P   Au5P   C�;eAuZ�   Au5P   Au�P   C�U�Au��   Au�P   AuP   C�].Au<�   AuP   Au�P   C�GUAu��   Au�P   Au�P   C�3WAu�   Au�P   AujP   C�6�Au��   AujP   Au�P   C�P�Au �   Au�P   AuLP   C�S�Auq�   AuLP   Au�P   C�O�Au��   Au�P   Au".P   C�S�Au"S�   Au".P   Au$�P   C�N4Au$��   Au$�P   Au'P   C�Q�Au'5�   Au'P   Au)�P   C�K5Au)��   Au)�P   Au+�P   C�FfAu,�   Au+�P   Au.cP   C�=`Au.��   Au.cP   Au0�P   C�P�Au0��   Au0�P   Au3EP   C�I�Au3j�   Au3EP   Au5�P   C�R:Au5��   Au5�P   Au8'P   C�S�Au8L�   Au8'P   Au:�P   C�U�Au:��   Au:�P   Au=	P   C�VKAu=.�   Au=	P   Au?zP   C�F�Au?��   Au?zP   AuA�P   C�@fAuB�   AuA�P   AuD\P   C�O~AuD��   AuD\P   AuF�P   C�W�AuF��   AuF�P   AuI>P   C�TPAuIc�   AuI>P   AuK�P   C�]~AuK��   AuK�P   AuN P   C�QkAuNE�   AuN P   AuP�P   C�LZAuP��   AuP�P   AuSP   C�/�AuS'�   AuSP   AuUsP   C�9�AuU��   AuUsP   AuW�P   C�ekAuX	�   AuW�P   AuZUP   C�k0AuZz�   AuZUP   Au\�P   C�LAu\��   Au\�P   Au_7P   C�G~Au_\�   Au_7P   Aua�P   C�O�Aua��   Aua�P   AudP   C�Z�Aud>�   AudP   Auf�P   C�[�Auf��   Auf�P   Auh�P   C�U�Aui �   Auh�P   AuklP   C�Y�Auk��   AuklP   Aum�P   C�^qAun�   Aum�P   AupNP   C�^�Aups�   AupNP   Aur�P   C�D`Aur��   Aur�P   Auu0P   C�M�AuuU�   Auu0P   Auw�P   C�U�Auw��   Auw�P   AuzP   C�R�Auz7�   AuzP   Au|�P   C�W�Au|��   Au|�P   Au~�P   C�YAu�   Au~�P   Au�eP   C�<�Au���   Au�eP   Au��P   C�<�Au���   Au��P   Au�GP   C�J�Au�l�   Au�GP   Au��P   C�8Au���   Au��P   Au�)P   C�9�Au�N�   Au�)P   Au��P   C�]�Au���   Au��P   Au�P   C�J�Au�0�   Au�P   Au�|P   C�S�Au���   Au�|P   Au��P   C�TdAu��   Au��P   Au�^P   C�J�Au���   Au�^P   Au��P   C�`�Au���   Au��P   Au�@P   C�_�Au�e�   Au�@P   Au��P   C�b�Au���   Au��P   Au�"P   C�G�Au�G�   Au�"P   Au��P   C�1Au���   Au��P   Au�P   C�`�Au�)�   Au�P   Au�uP   C�d�Au���   Au�uP   Au��P   C�D�Au��   Au��P   Au�WP   C�MAu�|�   Au�WP   Au��P   C�ZcAu���   Au��P   Au�9P   C�k�Au�^�   Au�9P   Au��P   C�b_Au���   Au��P   Au�P   C�b�Au�@�   Au�P   Au��P   C�^�Au���   Au��P   Au��P   C�f�Au�"�   Au��P   Au�nP   C�F�Au���   Au�nP   Au��P   C�m�Au��   Au��P   Au�PP   C�zYAu�u�   Au�PP   Au��P   C�CBAu���   Au��P   Au�2P   C�;�Au�W�   Au�2P   AuʣP   C�j�Au���   AuʣP   Au�P   C�Y�Au�9�   Au�P   AuυP   C�I0AuϪ�   AuυP   Au��P   C�REAu��   Au��P   Au�gP   C�J�AuԌ�   Au�gP   Au��P   C�^Au���   Au��P   Au�IP   C�VYAu�n�   Au�IP   AuۺP   C�PAu���   AuۺP   Au�+P   C�EAu�P�   Au�+P   Au��P   C�I�Au���   Au��P   Au�P   C�C&Au�2�   Au�P   Au�~P   C�AAu��   Au�~P   Au��P   C�OtAu��   Au��P   Au�`P   C�T�Au��   Au�`P   Au��P   C�WBAu���   Au��P   Au�BP   C�DHAu�g�   Au�BP   Au�P   C�S�