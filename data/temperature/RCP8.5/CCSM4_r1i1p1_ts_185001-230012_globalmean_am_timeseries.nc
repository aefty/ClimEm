CDF  �   
      time       bnds      lon       lat          %   CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 11:29:13 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//CCSM4_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//CCSM4_r1i1p1_ts_185001-230012_globalmean_am_timeseries.nc
Sun Feb 28 11:29:03 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//CCSM4_r1i1p1_ts_185001-230012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//CCSM4_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc
Sun Feb 28 11:28:49 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//CCSM4_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//CCSM4_r1i1p1_ts_185001-230012_fulldata.nc
Sat May 06 11:31:00 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:30:54 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/CCSM4/r1i1p1/ts_Amon_CCSM4_historical_r1i1p1_185001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
2011-10-21T17:19:45Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        CCSM4      institution       @NCAR (National Center for Atmospheric Research) Boulder, CO, USA   institute_id      NCAR   experiment_id         
historical     model_id      CCSM4      forcing       $Sl GHG Vl SS Ds SD BC MD OC Oz AA LU   parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @�H        contact       cesm_data@ucar.edu     
references        nGent P. R., et.al. 2011: The Community Climate System Model version 4. J. Climate, doi: 10.1175/2011JCLI4083.1     initialization_method               physics_version             tracking_id       $deb13cc9-f25d-44f9-a222-9f74b91ab964   acknowledgements     �The CESM project is supported by the National Science Foundation and the Office of Science (BER) of the U.S. Department of Energy. NCAR is sponsored by the National Science Foundation. Computing resources were provided by the Climate Simulation Laboratory at the NCAR Computational and Information Systems Laboratory (CISL), sponsored by the National Science Foundation and other agencies.      cesm_casename         b40.20th.track1.1deg.008   cesm_repotag      ccsm4_0_beta43     cesm_compset      B20TRCN    
resolution        f09_g16 (0.9x1.25_gx1v6)   forcing_note      �Additional information on the external forcings used in this experiment can be found at http://www.cesm.ucar.edu/CMIP5/forcing_information     processed_by      strandwg on mirage0 at 20111021    processing_code_information       �Last Changed Rev: 428 Last Changed Date: 2011-10-21 10:32:02 -0600 (Fri, 21 Oct 2011) Repository UUID: d2181dbe-5796-6825-dc7f-cbd98591f93d    product       output     
experiment        
historical     	frequency         year   creation_date         2011-10-21T17:19:45Z   
project_id        CMIP5      table_id      :Table Amon (26 July 2011) 976b7fd1d9e1be31dddd28f5dc79b7a1     title         0CCSM4 model output prepared for CMIP5 historical   parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.7.1      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T           t   	time_bnds                             |   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           d   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           l   ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       STS no change, CMIP5_table_comment: ""skin"" temperature (i.e., SST for open ocean)     original_name         TS     cell_methods      time: mean (interval: 30 days)     history       p2011-10-21T17:19:45Z altered by CMOR: replaced missing value flag (-1e+32) with standard missing value (1e+20).    associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CCSM4_historical_r0i0p0.nc areacella: areacella_fx_CCSM4_historical_r0i0p0.nc            �                Aq���   Aq��P   Aq�P   C���Aq�6�   Aq�P   Aq��P   C���Aq���   Aq��P   Aq��P   C��Aq��   Aq��P   Aq�dP   C���Aq���   Aq�dP   Aq��P   C���Aq���   Aq��P   Aq�FP   C��_Aq�k�   Aq�FP   Aq��P   C��dAq���   Aq��P   Aq�(P   C��Aq�M�   Aq�(P   Aq��P   C���Aq���   Aq��P   Aq�
P   C���Aq�/�   Aq�
P   Aq�{P   C���Aq���   Aq�{P   Aq��P   C��oAq��   Aq��P   Aq�]P   C���AqĂ�   Aq�]P   Aq��P   C���Aq���   Aq��P   Aq�?P   C���Aq�d�   Aq�?P   Aq˰P   C���Aq���   Aq˰P   Aq�!P   C��Aq�F�   Aq�!P   AqВP   C���Aqз�   AqВP   Aq�P   C���Aq�(�   Aq�P   Aq�tP   C���Aqՙ�   Aq�tP   Aq��P   C���Aq�
�   Aq��P   Aq�VP   C��OAq�{�   Aq�VP   Aq��P   C��7Aq���   Aq��P   Aq�8P   C��Aq�]�   Aq�8P   Aq�P   C���Aq���   Aq�P   Aq�P   C��{Aq�?�   Aq�P   Aq�P   C��{Aq��   Aq�P   Aq��P   C���Aq�!�   Aq��P   Aq�mP   C��fAq��   Aq�mP   Aq��P   C���Aq��   Aq��P   Aq�OP   C��)Aq�t�   Aq�OP   Aq��P   C���Aq���   Aq��P   Aq�1P   C��Aq�V�   Aq�1P   Aq��P   C���Aq���   Aq��P   Aq�P   C�fAq�8�   Aq�P   Aq��P   C���Aq���   Aq��P   Aq��P   C�v�Aq��   Aq��P   ArfP   C�lFAr��   ArfP   Ar�P   C��Ar��   Ar�P   ArHP   C��^Arm�   ArHP   Ar�P   C���Ar��   Ar�P   Ar*P   C��ArO�   Ar*P   Ar�P   C��Ar��   Ar�P   ArP   C���Ar1�   ArP   Ar}P   C��=Ar��   Ar}P   Ar�P   C��tAr�   Ar�P   Ar_P   C��|Ar��   Ar_P   Ar�P   C��*Ar��   Ar�P   ArAP   C���Arf�   ArAP   Ar�P   C��HAr��   Ar�P   Ar!#P   C���Ar!H�   Ar!#P   Ar#�P   C���Ar#��   Ar#�P   Ar&P   C���Ar&*�   Ar&P   Ar(vP   C�V�Ar(��   Ar(vP   Ar*�P   C�X�Ar+�   Ar*�P   Ar-XP   C�~�Ar-}�   Ar-XP   Ar/�P   C�xTAr/��   Ar/�P   Ar2:P   C�}_Ar2_�   Ar2:P   Ar4�P   C���Ar4��   Ar4�P   Ar7P   C��RAr7A�   Ar7P   Ar9�P   C��}Ar9��   Ar9�P   Ar;�P   C��HAr<#�   Ar;�P   Ar>oP   C���Ar>��   Ar>oP   Ar@�P   C���ArA�   Ar@�P   ArCQP   C��^ArCv�   ArCQP   ArE�P   C���ArE��   ArE�P   ArH3P   C��ArHX�   ArH3P   ArJ�P   C���ArJ��   ArJ�P   ArMP   C���ArM:�   ArMP   ArO�P   C��#ArO��   ArO�P   ArQ�P   C���ArR�   ArQ�P   ArThP   C���ArT��   ArThP   ArV�P   C��(ArV��   ArV�P   ArYJP   C��ArYo�   ArYJP   Ar[�P   C��OAr[��   Ar[�P   Ar^,P   C���Ar^Q�   Ar^,P   Ar`�P   C��rAr`��   Ar`�P   ArcP   C�ײArc3�   ArcP   AreP   C���Are��   AreP   Arg�P   C��8Arh�   Arg�P   ArjaP   C���Arj��   ArjaP   Arl�P   C�ҸArl��   Arl�P   AroCP   C��:Aroh�   AroCP   Arq�P   C���Arq��   Arq�P   Art%P   C��6ArtJ�   Art%P   Arv�P   C��;Arv��   Arv�P   AryP   C��?Ary,�   AryP   Ar{xP   C���Ar{��   Ar{xP   Ar}�P   C��	Ar~�   Ar}�P   Ar�ZP   C���Ar��   Ar�ZP   Ar��P   C�хAr���   Ar��P   Ar�<P   C�ʌAr�a�   Ar�<P   Ar��P   C���Ar���   Ar��P   Ar�P   C��Ar�C�   Ar�P   Ar��P   C��6Ar���   Ar��P   Ar� P   C��'Ar�%�   Ar� P   Ar�qP   C���Ar���   Ar�qP   Ar��P   C���Ar��   Ar��P   Ar�SP   C��Ar�x�   Ar�SP   Ar��P   C���Ar���   Ar��P   Ar�5P   C���Ar�Z�   Ar�5P   Ar��P   C�̅Ar���   Ar��P   Ar�P   C��tAr�<�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C��QAr��   Ar��P   Ar�jP   C�ƋAr���   Ar�jP   Ar��P   C�ΕAr� �   Ar��P   Ar�LP   C��vAr�q�   Ar�LP   Ar��P   C��|Ar���   Ar��P   Ar�.P   C��HAr�S�   Ar�.P   Ar��P   C��?Ar���   Ar��P   Ar�P   C���Ar�5�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar��P   C���Ar��   Ar��P   Ar�cP   C��%Ar���   Ar�cP   Ar��P   C���Ar���   Ar��P   Ar�EP   C���Ar�j�   Ar�EP   ArĶP   C���Ar���   ArĶP   Ar�'P   C��aAr�L�   Ar�'P   ArɘP   C��ZArɽ�   ArɘP   Ar�	P   C��2Ar�.�   Ar�	P   Ar�zP   C���ArΟ�   Ar�zP   Ar��P   C��RAr��   Ar��P   Ar�\P   C��(ArӁ�   Ar�\P   Ar��P   C��"Ar���   Ar��P   Ar�>P   C��Ar�c�   Ar�>P   ArگP   C��Ar���   ArگP   Ar� P   C��cAr�E�   Ar� P   ArߑP   C���Ar߶�   ArߑP   Ar�P   C���Ar�'�   Ar�P   Ar�sP   C���Ar��   Ar�sP   Ar��P   C� DAr�	�   Ar��P   Ar�UP   C��rAr�z�   Ar�UP   Ar��P   C��nAr���   Ar��P   Ar�7P   C��Ar�\�   Ar�7P   Ar�P   C��Ar���   Ar�P   Ar�P   C��	Ar�>�   Ar�P   Ar��P   C�?Ar���   Ar��P   Ar��P   C��Ar� �   Ar��P   Ar�lP   C���Ar���   Ar�lP   Ar��P   C��Ar��   Ar��P   Ar�NP   C��Ar�s�   Ar�NP   As�P   C���As��   As�P   As0P   C�	nAsU�   As0P   As�P   C��tAs��   As�P   As	P   C���As	7�   As	P   As�P   C��As��   As�P   As�P   C�As�   As�P   AseP   C�&�As��   AseP   As�P   C� �As��   As�P   AsGP   C�,GAsl�   AsGP   As�P   C�C^As��   As�P   As)P   C�DAsN�   As)P   As�P   C�(�As��   As�P   AsP   C�2�As0�   AsP   As!|P   C�=�As!��   As!|P   As#�P   C�A�As$�   As#�P   As&^P   C�@BAs&��   As&^P   As(�P   C�?�As(��   As(�P   As+@P   C�D�As+e�   As+@P   As-�P   C�N�As-��   As-�P   As0"P   C�;IAs0G�   As0"P   As2�P   C�7oAs2��   As2�P   As5P   C�;As5)�   As5P   As7uP   C�[�As7��   As7uP   As9�P   C�[�As:�   As9�P   As<WP   C�]_As<|�   As<WP   As>�P   C�_>As>��   As>�P   AsA9P   C�i�AsA^�   AsA9P   AsC�P   C�oAsC��   AsC�P   AsFP   C�W�AsF@�   AsFP   AsH�P   C�j\AsH��   AsH�P   AsJ�P   C�pfAsK"�   AsJ�P   AsMnP   C���AsM��   AsMnP   AsO�P   C�x�AsP�   AsO�P   AsRPP   C��[AsRu�   AsRPP   AsT�P   C���AsT��   AsT�P   AsW2P   C��yAsWW�   AsW2P   AsY�P   C���AsY��   AsY�P   As\P   C��As\9�   As\P   As^�P   C��KAs^��   As^�P   As`�P   C���Asa�   As`�P   AscgP   C���Asc��   AscgP   Ase�P   C���Ase��   Ase�P   AshIP   C��}Ashn�   AshIP   Asj�P   C���Asj��   Asj�P   Asm+P   C��	AsmP�   Asm+P   Aso�P   C���Aso��   Aso�P   AsrP   C��_Asr2�   AsrP   Ast~P   C���Ast��   Ast~P   Asv�P   C��#Asw�   Asv�P   Asy`P   C��1Asy��   Asy`P   As{�P   C��2As{��   As{�P   As~BP   C��QAs~g�   As~BP   As��P   C�ԵAs���   As��P   As�$P   C���As�I�   As�$P   As��P   C��As���   As��P   As�P   C�ӪAs�+�   As�P   As�wP   C��7As���   As�wP   As��P   C�ݮAs��   As��P   As�YP   C���As�~�   As�YP   As��P   C��8As���   As��P   As�;P   C���As�`�   As�;P   As��P   C��5As���   As��P   As�P   C��PAs�B�   As�P   As��P   C��As���   As��P   As��P   C�gAs�$�   As��P   As�pP   C�NAs���   As�pP   As��P   C��As��   As��P   As�RP   C�!gAs�w�   As�RP   As��P   C�&�As���   As��P   As�4P   C��As�Y�   As�4P   As��P   C�>�As���   As��P   As�P   C�D	As�;�   As�P   As��P   C�6)As���   As��P   As��P   C�F�As��   As��P   As�iP   C�K�As���   As�iP   As��P   C�NzAs���   As��P   As�KP   C�NLAs�p�   As�KP   As��P   C�\nAs���   As��P   As�-P   C�rHAs�R�   As�-P   AsP   C�iAs���   AsP   As�P   C�ZvAs�4�   As�P   AsǀP   C�W�Asǥ�   AsǀP   As��P   C�xzAs��   As��P   As�bP   C��CAṡ�   As�bP   As��P   C��fAs���   As��P   As�DP   C���As�i�   As�DP   AsӵP   C��?As���   AsӵP   As�&P   C��[As�K�   As�&P   AsؗP   C��XAsؼ�   AsؗP   As�P   C��As�-�   As�P   As�yP   C��cAsݞ�   As�yP   As��P   C���As��   As��P   As�[P   C���As��   As�[P   As��P   C���As���   As��P   As�=P   C��pAs�b�   As�=P   As�P   C�ȄAs���   As�P   As�P   C�ۦAs�D�   As�P   As�P   C���As��   As�P   As�P   C���As�&�   As�P   As�rP   C��XAs��   As�rP   As��P   C���As��   As��P   As�TP   C��kAs�y�   As�TP   As��P   C���As���   As��P   As�6P   C��@As�[�   As�6P   As��P   C���As���   As��P   AtP   C��At=�   AtP   At�P   C���At��   At�P   At�P   C�sAt�   At�P   At	kP   C�lAt	��   At	kP   At�P   C�&At�   At�P   AtMP   C��Atr�   AtMP   At�P   C�%At��   At�P   At/P   C�.@AtT�   At/P   At�P   C��At��   At�P   AtP   C��At6�   AtP   At�P   C��At��   At�P   At�P   C��At�   At�P   AtdP   C�"@At��   AtdP   At!�P   C�2�At!��   At!�P   At$FP   C�9pAt$k�   At$FP   At&�P   C�@�At&��   At&�P   At)(P   C�MPAt)M�   At)(P   At+�P   C�I�At+��   At+�P   At.
P   C�EwAt./�   At.
P   At0{P   C�GaAt0��   At0{P   At2�P   C�[bAt3�   At2�P   At5]P   C�W�At5��   At5]P   At7�P   C�gaAt7��   At7�P   At:?P   C�c�At:d�   At:?P   At<�P   C�f�At<��   At<�P   At?!P   C�_SAt?F�   At?!P   AtA�P   C�j#AtA��   AtA�P   AtDP   C�y�AtD(�   AtDP   AtFtP   C��IAtF��   AtFtP   AtH�P   C��)AtI
�   AtH�P   AtKVP   C�}qAtK{�   AtKVP   AtM�P   C��oAtM��   AtM�P   AtP8P   C��AtP]�   AtP8P   AtR�P   C��FAtR��   AtR�P   AtUP   C��AtU?�   AtUP   AtW�P   C��WAtW��   AtW�P   AtY�P   C��AtZ!�   AtY�P   At\mP   C���At\��   At\mP   At^�P   C�ÐAt_�   At^�P   AtaOP   C�ȷAtat�   AtaOP   Atc�P   C��XAtc��   Atc�P   Atf1P   C��AtfV�   Atf1P   Ath�P   C���Ath��   Ath�P   AtkP   C��hAtk8�   AtkP   Atm�P   C���Atm��   Atm�P   Ato�P   C��5Atp�   Ato�P   AtrfP   C��"Atr��   AtrfP   Att�P   C��NAtt��   Att�P   AtwHP   C�٠Atwm�   AtwHP   Aty�P   C���Aty��   Aty�P   At|*P   C���At|O�   At|*P   At~�P   C��*At~��   At~�P   At�P   C���At�1�   At�P   At�}P   C���At���   At�}P   At��P   C���At��   At��P   At�_P   C��	At���   At�_P   At��P   C��'At���   At��P   At�AP   C��<At�f�   At�AP   At��P   C���At���   At��P   At�#P   C���At�H�   At�#P   At��P   C��At���   At��P   At�P   C�At�*�   At�P   At�vP   C�WAt���   At�vP   At��P   C�wAt��   At��P   At�XP   C� �At�}�   At�XP   At��P   C��At���   At��P   At�:P   C�"At�_�   At�:P   At��P   C� At���   At��P   At�P   C�At�A�   At�P   At��P   C�RAt���   At��P   At��P   C�(<At�#�   At��P   At�oP   C�3�At���   At�oP   At��P   C�13At��   At��P   At�QP   C�1�At�v�   At�QP   At��P   C�9 At���   At��P   At�3P   C�4aAt�X�   At�3P   At��P   C�3�At���   At��P   At�P   C�E�At�:�   At�P   At��P   C�O�At���   At��P   At��P   C�D,At��   At��P   At�hP   C�D"Atō�   At�hP   At��P   C�\�At���   At��P   At�JP   C�UkAt�o�   At�JP   At̻P   C�T�At���   At̻P   At�,P   C�[�At�Q�   At�,P   AtѝP   C�d�At���   AtѝP   At�P   C�i�At�3�   At�P   At�P   C�p�At֤�   At�P   At��P   C�x�At��   At��P   At�aP   C�l$Atۆ�   At�aP   At��P   C�_�At���   At��P   At�CP   C�s�At�h�   At�CP   At�P   C���At���   At�P   At�%P   C��\At�J�   At�%P   At�P   C�|`At��   At�P   At�P   C�~�At�,�   At�P   At�xP   C��At��   At�xP   At��P   C��nAt��   At��P   At�ZP   C��ZAt��   At�ZP   At��P   C��)At���   At��P   At�<P   C���At�a�   At�<P   At��P   C���At���   At��P   At�P   C��EAt�C�   At�P   At��P   C���At���   At��P   Au  P   C��fAu %�   Au  P   AuqP   C���Au��   AuqP   Au�P   C��EAu�   Au�P   AuSP   C��1Aux�   AuSP   Au	�P   C���Au	��   Au	�P   Au5P   C���AuZ�   Au5P   Au�P   C��KAu��   Au�P   AuP   C��KAu<�   AuP   Au�P   C��DAu��   Au�P   Au�P   C���Au�   Au�P   AujP   C�ɺAu��   AujP   Au�P   C���Au �   Au�P   AuLP   C�¶Auq�   AuLP   Au�P   C�ʋAu��   Au�P   Au".P   C��NAu"S�   Au".P   Au$�P   C���Au$��   Au$�P   Au'P   C�� Au'5�   Au'P   Au)�P   C�ՔAu)��   Au)�P   Au+�P   C��VAu,�   Au+�P   Au.cP   C��Au.��   Au.cP   Au0�P   C���Au0��   Au0�P   Au3EP   C���Au3j�   Au3EP   Au5�P   C�ȀAu5��   Au5�P   Au8'P   C��Au8L�   Au8'P   Au:�P   C���Au:��   Au:�P   Au=	P   C���Au=.�   Au=	P   Au?zP   C��Au?��   Au?zP   AuA�P   C��AuB�   AuA�P   AuD\P   C��AuD��   AuD\P   AuF�P   C��FAuF��   AuF�P   AuI>P   C���AuIc�   AuI>P   AuK�P   C���AuK��   AuK�P   AuN P   C��AuNE�   AuN P   AuP�P   C��AuP��   AuP�P   AuSP   C���AuS'�   AuSP   AuUsP   C���AuU��   AuUsP   AuW�P   C��xAuX	�   AuW�P   AuZUP   C���AuZz�   AuZUP   Au\�P   C���Au\��   Au\�P   Au_7P   C���Au_\�   Au_7P   Aua�P   C�'Aua��   Aua�P   AudP   C��Aud>�   AudP   Auf�P   C�`Auf��   Auf�P   Auh�P   C���Aui �   Auh�P   AuklP   C�)Auk��   AuklP   Aum�P   C�5Aun�   Aum�P   AupNP   C��Aups�   AupNP   Aur�P   C�
�Aur��   Aur�P   Auu0P   C��AuuU�   Auu0P   Auw�P   C��Auw��   Auw�P   AuzP   C��Auz7�   AuzP   Au|�P   C�VAu|��   Au|�P   Au~�P   C��Au�   Au~�P   Au�eP   C��Au���   Au�eP   Au��P   C�Au���   Au��P   Au�GP   C�9Au�l�   Au�GP   Au��P   C�!�Au���   Au��P   Au�)P   C��Au�N�   Au�)P   Au��P   C�Au���   Au��P   Au�P   C�UAu�0�   Au�P   Au�|P   C��Au���   Au�|P   Au��P   C�"MAu��   Au��P   Au�^P   C�vAu���   Au�^P   Au��P   C�Au���   Au��P   Au�@P   C�)�Au�e�   Au�@P   Au��P   C�SAu���   Au��P   Au�"P   C��Au�G�   Au�"P   Au��P   C��Au���   Au��P   Au�P   C��Au�)�   Au�P   Au�uP   C��Au���   Au�uP   Au��P   C�!RAu��   Au��P   Au�WP   C��Au�|�   Au�WP   Au��P   C�'�Au���   Au��P   Au�9P   C�0rAu�^�   Au�9P   Au��P   C�(kAu���   Au��P   Au�P   C�57Au�@�   Au�P   Au��P   C�5�Au���   Au��P   Au��P   C�.�Au�"�   Au��P   Au�nP   C�9:Au���   Au�nP   Au��P   C�:�Au��   Au��P   Au�PP   C�,Au�u�   Au�PP   Au��P   C�>�Au���   Au��P   Au�2P   C�0RAu�W�   Au�2P   AuʣP   C�-uAu���   AuʣP   Au�P   C�2�Au�9�   Au�P   AuυP   C�0rAuϪ�   AuυP   Au��P   C�-�Au��   Au��P   Au�gP   C�:GAuԌ�   Au�gP   Au��P   C�>$Au���   Au��P   Au�IP   C�DAu�n�   Au�IP   AuۺP   C�:�Au���   AuۺP   Au�+P   C�5Au�P�   Au�+P   Au��P   C�=�Au���   Au��P   Au�P   C�?�Au�2�   Au�P   Au�~P   C�;lAu��   Au�~P   Au��P   C�=�Au��   Au��P   Au�`P   C�E�Au��   Au�`P   Au��P   C�A]Au���   Au��P   Au�BP   C�>�Au�g�   Au�BP   Au�P   C�F<