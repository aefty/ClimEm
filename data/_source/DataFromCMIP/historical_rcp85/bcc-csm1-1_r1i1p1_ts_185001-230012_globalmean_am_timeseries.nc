CDF  �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 11:27:55 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//bcc-csm1-1_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//bcc-csm1-1_r1i1p1_ts_185001-230012_globalmean_am_timeseries.nc
Sun Feb 28 11:27:53 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//bcc-csm1-1_r1i1p1_ts_185001-230012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//bcc-csm1-1_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc
Sun Feb 28 11:27:50 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//bcc-csm1-1_r1i1p1_ts_185001-201212_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//bcc-csm1-1_r1i1p1_ts_185001-230012_fulldata.nc
Sat May 06 11:39:52 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:39:51 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/bcc-csm1-1/r1i1p1/ts_Amon_bcc-csm1-1_historical_r1i1p1_185001-201212.nc /echam/folini/cmip5/historical//tmp_01.nc
Output from monthly mean data 2011-06-15T08:55:23Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.   source        �bcc-csm1-1:atmosphere:  BCC_AGCM2.1 (T42L26); land: BCC_AVIM1.0;ocean: MOM4_L40 (tripolar, 1 lon x (1-1/3) lat, L40);sea ice: SIS (tripolar,1 lon x (1-1/3) lat)   institution       EBeijing Climate Center(BCC),China Meteorological Administration,China      institute_id      BCC    experiment_id         
historical     model_id      
bcc-csm1-1     forcing       #Nat Ant GHG SD Oz Sl Vl SS Ds BC OC    parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @}`        contact        Dr. Tongwen Wu (twwu@cma.gov.cn)   comment       mThe experiment starts from piControl run at year 470. RCP8.5 scenario forcing data are used beyond year 2005.      initialization_method               physics_version             tracking_id       $f807e04d-d396-475e-afca-32c7c8271216   product       output     
experiment        
historical     	frequency         year   creation_date         2011-06-15T08:55:24Z   
project_id        CMIP5      table_id      ;Table Amon (11 April 2011) 1cfdc7322cf2f4a32614826fab42c1ab    title         5bcc-csm1-1 model output prepared for CMIP5 historical      parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.5.6      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T           �   	time_bnds                             �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   ts                     	   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         TS     cell_methods      "time: mean (interval: 20 mintues)      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_bcc-csm1-1_historical_r0i0p0.nc areacella: areacella_fx_bcc-csm1-1_historical_r0i0p0.nc                          Aq���   Aq��P   Aq�P   C��jAq�6�   Aq�P   Aq��P   C�vWAq���   Aq��P   Aq��P   C�w�Aq��   Aq��P   Aq�dP   C���Aq���   Aq�dP   Aq��P   C���Aq���   Aq��P   Aq�FP   C��NAq�k�   Aq�FP   Aq��P   C�{gAq���   Aq��P   Aq�(P   C�XAq�M�   Aq�(P   Aq��P   C�l�Aq���   Aq��P   Aq�
P   C�Aq�/�   Aq�
P   Aq�{P   C��)Aq���   Aq�{P   Aq��P   C���Aq��   Aq��P   Aq�]P   C���AqĂ�   Aq�]P   Aq��P   C�nAq���   Aq��P   Aq�?P   C���Aq�d�   Aq�?P   Aq˰P   C���Aq���   Aq˰P   Aq�!P   C�|Aq�F�   Aq�!P   AqВP   C��
Aqз�   AqВP   Aq�P   C���Aq�(�   Aq�P   Aq�tP   C��%Aqՙ�   Aq�tP   Aq��P   C��Aq�
�   Aq��P   Aq�VP   C��Aq�{�   Aq�VP   Aq��P   C���Aq���   Aq��P   Aq�8P   C���Aq�]�   Aq�8P   Aq�P   C���Aq���   Aq�P   Aq�P   C��/Aq�?�   Aq�P   Aq�P   C���Aq��   Aq�P   Aq��P   C��{Aq�!�   Aq��P   Aq�mP   C���Aq��   Aq�mP   Aq��P   C��]Aq��   Aq��P   Aq�OP   C���Aq�t�   Aq�OP   Aq��P   C���Aq���   Aq��P   Aq�1P   C��!Aq�V�   Aq�1P   Aq��P   C��zAq���   Aq��P   Aq�P   C�[!Aq�8�   Aq�P   Aq��P   C�x�Aq���   Aq��P   Aq��P   C�t�Aq��   Aq��P   ArfP   C��vAr��   ArfP   Ar�P   C�|rAr��   Ar�P   ArHP   C�w�Arm�   ArHP   Ar�P   C��Ar��   Ar�P   Ar*P   C�f�ArO�   Ar*P   Ar�P   C�v�Ar��   Ar�P   ArP   C���Ar1�   ArP   Ar}P   C��WAr��   Ar}P   Ar�P   C��>Ar�   Ar�P   Ar_P   C���Ar��   Ar_P   Ar�P   C���Ar��   Ar�P   ArAP   C���Arf�   ArAP   Ar�P   C���Ar��   Ar�P   Ar!#P   C���Ar!H�   Ar!#P   Ar#�P   C��VAr#��   Ar#�P   Ar&P   C��FAr&*�   Ar&P   Ar(vP   C�g�Ar(��   Ar(vP   Ar*�P   C�g(Ar+�   Ar*�P   Ar-XP   C�tFAr-}�   Ar-XP   Ar/�P   C���Ar/��   Ar/�P   Ar2:P   C���Ar2_�   Ar2:P   Ar4�P   C�{�Ar4��   Ar4�P   Ar7P   C��JAr7A�   Ar7P   Ar9�P   C�w�Ar9��   Ar9�P   Ar;�P   C�z�Ar<#�   Ar;�P   Ar>oP   C�}[Ar>��   Ar>oP   Ar@�P   C�w`ArA�   Ar@�P   ArCQP   C��GArCv�   ArCQP   ArE�P   C��hArE��   ArE�P   ArH3P   C���ArHX�   ArH3P   ArJ�P   C��ArJ��   ArJ�P   ArMP   C��8ArM:�   ArMP   ArO�P   C���ArO��   ArO�P   ArQ�P   C��	ArR�   ArQ�P   ArThP   C��9ArT��   ArThP   ArV�P   C���ArV��   ArV�P   ArYJP   C��@ArYo�   ArYJP   Ar[�P   C��HAr[��   Ar[�P   Ar^,P   C���Ar^Q�   Ar^,P   Ar`�P   C���Ar`��   Ar`�P   ArcP   C��+Arc3�   ArcP   AreP   C���Are��   AreP   Arg�P   C���Arh�   Arg�P   ArjaP   C���Arj��   ArjaP   Arl�P   C���Arl��   Arl�P   AroCP   C���Aroh�   AroCP   Arq�P   C���Arq��   Arq�P   Art%P   C��eArtJ�   Art%P   Arv�P   C���Arv��   Arv�P   AryP   C��Ary,�   AryP   Ar{xP   C��"Ar{��   Ar{xP   Ar}�P   C��$Ar~�   Ar}�P   Ar�ZP   C��DAr��   Ar�ZP   Ar��P   C��Ar���   Ar��P   Ar�<P   C���Ar�a�   Ar�<P   Ar��P   C���Ar���   Ar��P   Ar�P   C��Ar�C�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar� P   C��Ar�%�   Ar� P   Ar�qP   C��Ar���   Ar�qP   Ar��P   C��}Ar��   Ar��P   Ar�SP   C��|Ar�x�   Ar�SP   Ar��P   C��JAr���   Ar��P   Ar�5P   C���Ar�Z�   Ar�5P   Ar��P   C���Ar���   Ar��P   Ar�P   C��hAr�<�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C��oAr��   Ar��P   Ar�jP   C���Ar���   Ar�jP   Ar��P   C��3Ar� �   Ar��P   Ar�LP   C���Ar�q�   Ar�LP   Ar��P   C��"Ar���   Ar��P   Ar�.P   C���Ar�S�   Ar�.P   Ar��P   C���Ar���   Ar��P   Ar�P   C��Ar�5�   Ar�P   Ar��P   C��&Ar���   Ar��P   Ar��P   C���Ar��   Ar��P   Ar�cP   C���Ar���   Ar�cP   Ar��P   C��Ar���   Ar��P   Ar�EP   C���Ar�j�   Ar�EP   ArĶP   C��~Ar���   ArĶP   Ar�'P   C���Ar�L�   Ar�'P   ArɘP   C��GArɽ�   ArɘP   Ar�	P   C���Ar�.�   Ar�	P   Ar�zP   C��+ArΟ�   Ar�zP   Ar��P   C��NAr��   Ar��P   Ar�\P   C��XArӁ�   Ar�\P   Ar��P   C��
Ar���   Ar��P   Ar�>P   C��qAr�c�   Ar�>P   ArگP   C���Ar���   ArگP   Ar� P   C��KAr�E�   Ar� P   ArߑP   C���Ar߶�   ArߑP   Ar�P   C���Ar�'�   Ar�P   Ar�sP   C��:Ar��   Ar�sP   Ar��P   C���Ar�	�   Ar��P   Ar�UP   C��rAr�z�   Ar�UP   Ar��P   C���Ar���   Ar��P   Ar�7P   C�խAr�\�   Ar�7P   Ar�P   C��>Ar���   Ar�P   Ar�P   C��MAr�>�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar��P   C��
Ar� �   Ar��P   Ar�lP   C��Ar���   Ar�lP   Ar��P   C��Ar��   Ar��P   Ar�NP   C�	Ar�s�   Ar�NP   As�P   C���As��   As�P   As0P   C��AsU�   As0P   As�P   C���As��   As�P   As	P   C��jAs	7�   As	P   As�P   C�	�As��   As�P   As�P   C��As�   As�P   AseP   C�TAs��   AseP   As�P   C��As��   As�P   AsGP   C��Asl�   AsGP   As�P   C�OAs��   As�P   As)P   C�#wAsN�   As)P   As�P   C�As��   As�P   AsP   C��As0�   AsP   As!|P   C��As!��   As!|P   As#�P   C�)0As$�   As#�P   As&^P   C�7?As&��   As&^P   As(�P   C�A-As(��   As(�P   As+@P   C�;�As+e�   As+@P   As-�P   C�=\As-��   As-�P   As0"P   C�>*As0G�   As0"P   As2�P   C�.zAs2��   As2�P   As5P   C�7BAs5)�   As5P   As7uP   C�/2As7��   As7uP   As9�P   C�8�As:�   As9�P   As<WP   C�7|As<|�   As<WP   As>�P   C�D�As>��   As>�P   AsA9P   C�M AsA^�   AsA9P   AsC�P   C�[SAsC��   AsC�P   AsFP   C�Z(AsF@�   AsFP   AsH�P   C�CiAsH��   AsH�P   AsJ�P   C�@AsK"�   AsJ�P   AsMnP   C�a�AsM��   AsMnP   AsO�P   C�ZzAsP�   AsO�P   AsRPP   C�OAsRu�   AsRPP   AsT�P   C�]AsT��   AsT�P   AsW2P   C�g_AsWW�   AsW2P   AsY�P   C�k�AsY��   AsY�P   As\P   C�f|As\9�   As\P   As^�P   C�mAs^��   As^�P   As`�P   C�b#Asa�   As`�P   AscgP   C�p�Asc��   AscgP   Ase�P   C�v Ase��   Ase�P   AshIP   C���Ashn�   AshIP   Asj�P   C���Asj��   Asj�P   Asm+P   C��.AsmP�   Asm+P   Aso�P   C��NAso��   Aso�P   AsrP   C���Asr2�   AsrP   Ast~P   C���Ast��   Ast~P   Asv�P   C��Asw�   Asv�P   Asy`P   C���Asy��   Asy`P   As{�P   C���As{��   As{�P   As~BP   C��As~g�   As~BP   As��P   C��#As���   As��P   As�$P   C���As�I�   As�$P   As��P   C��5As���   As��P   As�P   C��	As�+�   As�P   As�wP   C��YAs���   As�wP   As��P   C���As��   As��P   As�YP   C�єAs�~�   As�YP   As��P   C��:As���   As��P   As�;P   C��}As�`�   As�;P   As��P   C���As���   As��P   As�P   C���As�B�   As�P   As��P   C��.As���   As��P   As��P   C��As�$�   As��P   As�pP   C��=As���   As�pP   As��P   C���As��   As��P   As�RP   C�
�As�w�   As�RP   As��P   C�}As���   As��P   As�4P   C�dAs�Y�   As�4P   As��P   C��As���   As��P   As�P   C�6�As�;�   As�P   As��P   C�9�As���   As��P   As��P   C�;�As��   As��P   As�iP   C�O�As���   As�iP   As��P   C�SAs���   As��P   As�KP   C�@As�p�   As�KP   As��P   C�D�As���   As��P   As�-P   C�J�As�R�   As�-P   AsP   C�O�As���   AsP   As�P   C�\�As�4�   As�P   AsǀP   C�ZAsǥ�   AsǀP   As��P   C�J�As��   As��P   As�bP   C�U7Aṡ�   As�bP   As��P   C�lUAs���   As��P   As�DP   C�k>As�i�   As�DP   AsӵP   C�h;As���   AsӵP   As�&P   C�w@As�K�   As�&P   AsؗP   C�xAsؼ�   AsؗP   As�P   C���As�-�   As�P   As�yP   C���Asݞ�   As�yP   As��P   C��#As��   As��P   As�[P   C��0As��   As�[P   As��P   C�� As���   As��P   As�=P   C��~As�b�   As�=P   As�P   C���As���   As�P   As�P   C�� As�D�   As�P   As�P   C��GAs��   As�P   As�P   C��@As�&�   As�P   As�rP   C��tAs��   As�rP   As��P   C���As��   As��P   As�TP   C�ȡAs�y�   As�TP   As��P   C���As���   As��P   As�6P   C�ԽAs�[�   As�6P   As��P   C�ʍAs���   As��P   AtP   C���At=�   AtP   At�P   C��PAt��   At�P   At�P   C��At�   At�P   At	kP   C��FAt	��   At	kP   At�P   C��;At�   At�P   AtMP   C���Atr�   AtMP   At�P   C��gAt��   At�P   At/P   C�
AtT�   At/P   At�P   C�OAt��   At�P   AtP   C�, At6�   AtP   At�P   C��At��   At�P   At�P   C�2At�   At�P   AtdP   C��At��   AtdP   At!�P   C�">At!��   At!�P   At$FP   C�0TAt$k�   At$FP   At&�P   C�3XAt&��   At&�P   At)(P   C�yAt)M�   At)(P   At+�P   C�5�At+��   At+�P   At.
P   C�7At./�   At.
P   At0{P   C�1�At0��   At0{P   At2�P   C�>[At3�   At2�P   At5]P   C�N�At5��   At5]P   At7�P   C�RAt7��   At7�P   At:?P   C�YAt:d�   At:?P   At<�P   C�^�At<��   At<�P   At?!P   C�eZAt?F�   At?!P   AtA�P   C�`SAtA��   AtA�P   AtDP   C�m�AtD(�   AtDP   AtFtP   C�yAtF��   AtFtP   AtH�P   C�ytAtI
�   AtH�P   AtKVP   C��AtK{�   AtKVP   AtM�P   C��BAtM��   AtM�P   AtP8P   C��AtP]�   AtP8P   AtR�P   C��ZAtR��   AtR�P   AtUP   C���AtU?�   AtUP   AtW�P   C��tAtW��   AtW�P   AtY�P   C��nAtZ!�   AtY�P   At\mP   C���At\��   At\mP   At^�P   C���At_�   At^�P   AtaOP   C��'Atat�   AtaOP   Atc�P   C��ZAtc��   Atc�P   Atf1P   C���AtfV�   Atf1P   Ath�P   C���Ath��   Ath�P   AtkP   C���Atk8�   AtkP   Atm�P   C��Atm��   Atm�P   Ato�P   C���Atp�   Ato�P   AtrfP   C���Atr��   AtrfP   Att�P   C���Att��   Att�P   AtwHP   C���Atwm�   AtwHP   Aty�P   C�ӫAty��   Aty�P   At|*P   C��\At|O�   At|*P   At~�P   C��lAt~��   At~�P   At�P   C��At�1�   At�P   At�}P   C�ȕAt���   At�}P   At��P   C���At��   At��P   At�_P   C���At���   At�_P   At��P   C��"At���   At��P   At�AP   C��At�f�   At�AP   At��P   C���At���   At��P   At�#P   C���At�H�   At�#P   At��P   C��+At���   At��P   At�P   C��CAt�*�   At�P   At�vP   C��^At���   At�vP   At��P   C� �At��   At��P   At�XP   C��At�}�   At�XP   At��P   C��At���   At��P   At�:P   C�At�_�   At�:P   At��P   C��At���   At��P   At�P   C��At�A�   At�P   At��P   C��At���   At��P   At��P   C�zAt�#�   At��P   At�oP   C�SAt���   At�oP   At��P   C�.�At��   At��P   At�QP   C�'�At�v�   At�QP   At��P   C�/SAt���   At��P   At�3P   C�A�At�X�   At�3P   At��P   C�@lAt���   At��P   At�P   C�/�At�:�   At�P   At��P   C�?JAt���   At��P   At��P   C�E*At��   At��P   At�hP   C�L�Atō�   At�hP   At��P   C�C_At���   At��P   At�JP   C�G�At�o�   At�JP   At̻P   C�Q5At���   At̻P   At�,P   C�]�At�Q�   At�,P   AtѝP   C�hAt���   AtѝP   At�P   C�\�At�3�   At�P   At�P   C�`WAt֤�   At�P   At��P   C�[�At��   At��P   At�aP   C�U~Atۆ�   At�aP   At��P   C�p�At���   At��P   At�CP   C�nAt�h�   At�CP   At�P   C�m:At���   At�P   At�%P   C�x�At�J�   At�%P   At�P   C�ycAt��   At�P   At�P   C�z6At�,�   At�P   At�xP   C�|�At��   At�xP   At��P   C�~@At��   At��P   At�ZP   C�~eAt��   At�ZP   At��P   C�~�At���   At��P   At�<P   C���At�a�   At�<P   At��P   C��RAt���   At��P   At�P   C���At�C�   At�P   At��P   C��3At���   At��P   Au  P   C��[Au %�   Au  P   AuqP   C��Au��   AuqP   Au�P   C��UAu�   Au�P   AuSP   C��SAux�   AuSP   Au	�P   C���Au	��   Au	�P   Au5P   C���AuZ�   Au5P   Au�P   C��Au��   Au�P   AuP   C��4Au<�   AuP   Au�P   C��kAu��   Au�P   Au�P   C��MAu�   Au�P   AujP   C���Au��   AujP   Au�P   C���Au �   Au�P   AuLP   C���Auq�   AuLP   Au�P   C��Au��   Au�P   Au".P   C���Au"S�   Au".P   Au$�P   C��ZAu$��   Au$�P   Au'P   C��Au'5�   Au'P   Au)�P   C�� Au)��   Au)�P   Au+�P   C���Au,�   Au+�P   Au.cP   C��aAu.��   Au.cP   Au0�P   C��NAu0��   Au0�P   Au3EP   C���Au3j�   Au3EP   Au5�P   C���Au5��   Au5�P   Au8'P   C���Au8L�   Au8'P   Au:�P   C�ɩAu:��   Au:�P   Au=	P   C�̫Au=.�   Au=	P   Au?zP   C�ŭAu?��   Au?zP   AuA�P   C��&AuB�   AuA�P   AuD\P   C��}AuD��   AuD\P   AuF�P   C�رAuF��   AuF�P   AuI>P   C�҇AuIc�   AuI>P   AuK�P   C�ĈAuK��   AuK�P   AuN P   C�ՑAuNE�   AuN P   AuP�P   C��AuP��   AuP�P   AuSP   C���AuS'�   AuSP   AuUsP   C��AuU��   AuUsP   AuW�P   C�ܖAuX	�   AuW�P   AuZUP   C���AuZz�   AuZUP   Au\�P   C��Au\��   Au\�P   Au_7P   C��JAu_\�   Au_7P   Aua�P   C��nAua��   Aua�P   AudP   C���Aud>�   AudP   Auf�P   C���Auf��   Auf�P   Auh�P   C��Aui �   Auh�P   AuklP   C��Auk��   AuklP   Aum�P   C��PAun�   Aum�P   AupNP   C��.Aups�   AupNP   Aur�P   C��Aur��   Aur�P   Auu0P   C���AuuU�   Auu0P   Auw�P   C��TAuw��   Auw�P   AuzP   C��VAuz7�   AuzP   Au|�P   C��^Au|��   Au|�P   Au~�P   C��$Au�   Au~�P   Au�eP   C���Au���   Au�eP   Au��P   C���Au���   Au��P   Au�GP   C��Au�l�   Au�GP   Au��P   C��SAu���   Au��P   Au�)P   C��aAu�N�   Au�)P   Au��P   C��pAu���   Au��P   Au�P   C���Au�0�   Au�P   Au�|P   C�_Au���   Au�|P   Au��P   C��Au��   Au��P   Au�^P   C� �Au���   Au�^P   Au��P   C� �Au���   Au��P   Au�@P   C���Au�e�   Au�@P   Au��P   C� JAu���   Au��P   Au�"P   C��Au�G�   Au�"P   Au��P   C���Au���   Au��P   Au�P   C���Au�)�   Au�P   Au�uP   C��Au���   Au�uP   Au��P   C�Au��   Au��P   Au�WP   C��Au�|�   Au�WP   Au��P   C�iAu���   Au��P   Au�9P   C� ,Au�^�   Au�9P   Au��P   C�
�Au���   Au��P   Au�P   C��pAu�@�   Au�P   Au��P   C��4Au���   Au��P   Au��P   C�Au�"�   Au��P   Au�nP   C��Au���   Au�nP   Au��P   C��?Au��   Au��P   Au�PP   C��Au�u�   Au�PP   Au��P   C�	�Au���   Au��P   Au�2P   C��Au�W�   Au�2P   AuʣP   C�aAu���   AuʣP   Au�P   C�Au�9�   Au�P   AuυP   C��AuϪ�   AuυP   Au��P   C�`Au��   Au��P   Au�gP   C��AuԌ�   Au�gP   Au��P   C��Au���   Au��P   Au�IP   C��Au�n�   Au�IP   AuۺP   C�hAu���   AuۺP   Au�+P   C��Au�P�   Au�+P   Au��P   C�}Au���   Au��P   Au�P   C��Au�2�   Au�P   Au�~P   C�2Au��   Au�~P   Au��P   C�`Au��   Au��P   Au�`P   C��Au��   Au�`P   Au��P   C��Au���   Au��P   Au�BP   C�pAu�g�   Au�BP   Au�P   C�|