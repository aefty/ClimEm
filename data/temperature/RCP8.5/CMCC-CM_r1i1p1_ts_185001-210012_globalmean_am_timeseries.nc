CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      Sun Feb 28 11:33:08 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//CMCC-CM_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//CMCC-CM_r1i1p1_ts_185001-210012_globalmean_am_timeseries.nc
Sun Feb 28 11:32:56 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//CMCC-CM_r1i1p1_ts_185001-210012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//CMCC-CM_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc
Sun Feb 28 11:32:40 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//CMCC-CM_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//CMCC-CM_r1i1p1_ts_185001-210012_fulldata.nc
Sat May 06 11:32:49 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:32:34 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_185001-185912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_186001-186912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_187001-187912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_188001-188912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_189001-189912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_190001-190912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_191001-191912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_192001-192912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_193001-193912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_194001-194912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_195001-195912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_196001-196912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_197001-197912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_198001-198912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_199001-199912.nc /net/atmos/data/cmip5/historical/Amon/ts/CMCC-CM/r1i1p1/ts_Amon_CMCC-CM_historical_r1i1p1_200001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
Model output postprocessed with CDO 2011-11-30T13:25:05Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.    source        CMCC-CM    institution       1CMCC - Centro Euro-Mediterraneo per i Cambiamenti      institute_id      CMCC   experiment_id         
historical     model_id      CMCC-CM    forcing       Nat,Ant,GHG,SA,TO,Sl   parent_experiment_id      	piControl      parent_experiment_rip         N/A    branch_time       @���       contact       !Silvio Gualdi (gualdi@bo.ingv.it)      comment       ~simulation starting at the end of the piControl run, thus after 600+300=900 years spin-up at pre-industrial GHG concentrations     
references        Mmodel described in the documentation at http://www.cmcc.it/data-models/models      initialization_method               physics_version             tracking_id       $73c8d6df-5b90-4da3-9061-e82fbea7ca1f   product       output     
experiment        
historical     	frequency         year   creation_date         2011-11-30T13:25:05Z   
project_id        CMIP5      table_id      ;Table Amon (27 April 2011) a5a1c518f52ae340313ba0aada03f862    title         2CMCC-CM model output prepared for CMIP5 historical     parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.7.1      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      standard   axis      T           $   	time_bnds                             ,   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X              lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y              ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         tsurf      cell_methods      time: mean (interval: 1 month)     associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CMCC-CM_historical_r0i0p0.nc areacella: areacella_fx_CMCC-CM_historical_r0i0p0.nc       history       :2011-11-30T13:25:05Z altered by CMOR: Inverted axis: lat.           <                Aq���   Aq��P   Aq�P   C���Aq�6�   Aq�P   Aq��P   C���Aq���   Aq��P   Aq��P   C��FAq��   Aq��P   Aq�dP   C���Aq���   Aq�dP   Aq��P   C�w\Aq���   Aq��P   Aq�FP   C�~�Aq�k�   Aq�FP   Aq��P   C��Aq���   Aq��P   Aq�(P   C��.Aq�M�   Aq�(P   Aq��P   C�~�Aq���   Aq��P   Aq�
P   C��EAq�/�   Aq�
P   Aq�{P   C���Aq���   Aq�{P   Aq��P   C���Aq��   Aq��P   Aq�]P   C��AqĂ�   Aq�]P   Aq��P   C�{Aq���   Aq��P   Aq�?P   C��tAq�d�   Aq�?P   Aq˰P   C�}�Aq���   Aq˰P   Aq�!P   C�}3Aq�F�   Aq�!P   AqВP   C���Aqз�   AqВP   Aq�P   C��UAq�(�   Aq�P   Aq�tP   C���Aqՙ�   Aq�tP   Aq��P   C�{0Aq�
�   Aq��P   Aq�VP   C��Aq�{�   Aq�VP   Aq��P   C��0Aq���   Aq��P   Aq�8P   C���Aq�]�   Aq�8P   Aq�P   C��?Aq���   Aq�P   Aq�P   C���Aq�?�   Aq�P   Aq�P   C��Aq��   Aq�P   Aq��P   C�~Aq�!�   Aq��P   Aq�mP   C�eKAq��   Aq�mP   Aq��P   C�|�Aq��   Aq��P   Aq�OP   C���Aq�t�   Aq�OP   Aq��P   C��YAq���   Aq��P   Aq�1P   C��:Aq�V�   Aq�1P   Aq��P   C�n�Aq���   Aq��P   Aq�P   C�~�Aq�8�   Aq�P   Aq��P   C��=Aq���   Aq��P   Aq��P   C��;Aq��   Aq��P   ArfP   C�� Ar��   ArfP   Ar�P   C�� Ar��   Ar�P   ArHP   C��cArm�   ArHP   Ar�P   C��;Ar��   Ar�P   Ar*P   C���ArO�   Ar*P   Ar�P   C��8Ar��   Ar�P   ArP   C�zAr1�   ArP   Ar}P   C�v�Ar��   Ar}P   Ar�P   C��*Ar�   Ar�P   Ar_P   C��oAr��   Ar_P   Ar�P   C��?Ar��   Ar�P   ArAP   C��tArf�   ArAP   Ar�P   C��Ar��   Ar�P   Ar!#P   C��:Ar!H�   Ar!#P   Ar#�P   C��cAr#��   Ar#�P   Ar&P   C���Ar&*�   Ar&P   Ar(vP   C���Ar(��   Ar(vP   Ar*�P   C���Ar+�   Ar*�P   Ar-XP   C��Ar-}�   Ar-XP   Ar/�P   C��(Ar/��   Ar/�P   Ar2:P   C��Ar2_�   Ar2:P   Ar4�P   C��Ar4��   Ar4�P   Ar7P   C���Ar7A�   Ar7P   Ar9�P   C���Ar9��   Ar9�P   Ar;�P   C��Ar<#�   Ar;�P   Ar>oP   C���Ar>��   Ar>oP   Ar@�P   C���ArA�   Ar@�P   ArCQP   C�_ArCv�   ArCQP   ArE�P   C���ArE��   ArE�P   ArH3P   C���ArHX�   ArH3P   ArJ�P   C�vXArJ��   ArJ�P   ArMP   C�s�ArM:�   ArMP   ArO�P   C��~ArO��   ArO�P   ArQ�P   C���ArR�   ArQ�P   ArThP   C���ArT��   ArThP   ArV�P   C���ArV��   ArV�P   ArYJP   C�w�ArYo�   ArYJP   Ar[�P   C��rAr[��   Ar[�P   Ar^,P   C���Ar^Q�   Ar^,P   Ar`�P   C��DAr`��   Ar`�P   ArcP   C���Arc3�   ArcP   AreP   C��Are��   AreP   Arg�P   C���Arh�   Arg�P   ArjaP   C��sArj��   ArjaP   Arl�P   C���Arl��   Arl�P   AroCP   C��cAroh�   AroCP   Arq�P   C��5Arq��   Arq�P   Art%P   C���ArtJ�   Art%P   Arv�P   C���Arv��   Arv�P   AryP   C��&Ary,�   AryP   Ar{xP   C���Ar{��   Ar{xP   Ar}�P   C���Ar~�   Ar}�P   Ar�ZP   C���Ar��   Ar�ZP   Ar��P   C���Ar���   Ar��P   Ar�<P   C��SAr�a�   Ar�<P   Ar��P   C��+Ar���   Ar��P   Ar�P   C��|Ar�C�   Ar�P   Ar��P   C��LAr���   Ar��P   Ar� P   C���Ar�%�   Ar� P   Ar�qP   C��Ar���   Ar�qP   Ar��P   C���Ar��   Ar��P   Ar�SP   C��!Ar�x�   Ar�SP   Ar��P   C���Ar���   Ar��P   Ar�5P   C��<Ar�Z�   Ar�5P   Ar��P   C��rAr���   Ar��P   Ar�P   C���Ar�<�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar��P   C��Ar��   Ar��P   Ar�jP   C��!Ar���   Ar�jP   Ar��P   C��]Ar� �   Ar��P   Ar�LP   C���Ar�q�   Ar�LP   Ar��P   C��Ar���   Ar��P   Ar�.P   C���Ar�S�   Ar�.P   Ar��P   C�}�Ar���   Ar��P   Ar�P   C��FAr�5�   Ar�P   Ar��P   C��DAr���   Ar��P   Ar��P   C��Ar��   Ar��P   Ar�cP   C���Ar���   Ar�cP   Ar��P   C���Ar���   Ar��P   Ar�EP   C��PAr�j�   Ar�EP   ArĶP   C���Ar���   ArĶP   Ar�'P   C��yAr�L�   Ar�'P   ArɘP   C���Arɽ�   ArɘP   Ar�	P   C���Ar�.�   Ar�	P   Ar�zP   C��YArΟ�   Ar�zP   Ar��P   C��RAr��   Ar��P   Ar�\P   C���ArӁ�   Ar�\P   Ar��P   C��Ar���   Ar��P   Ar�>P   C��9Ar�c�   Ar�>P   ArگP   C���Ar���   ArگP   Ar� P   C��5Ar�E�   Ar� P   ArߑP   C��Ar߶�   ArߑP   Ar�P   C���Ar�'�   Ar�P   Ar�sP   C���Ar��   Ar�sP   Ar��P   C���Ar�	�   Ar��P   Ar�UP   C�� Ar�z�   Ar�UP   Ar��P   C��rAr���   Ar��P   Ar�7P   C���Ar�\�   Ar�7P   Ar�P   C���Ar���   Ar�P   Ar�P   C���Ar�>�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C��+Ar� �   Ar��P   Ar�lP   C���Ar���   Ar�lP   Ar��P   C���Ar��   Ar��P   Ar�NP   C��(Ar�s�   Ar�NP   As�P   C���As��   As�P   As0P   C��FAsU�   As0P   As�P   C��zAs��   As�P   As	P   C��KAs	7�   As	P   As�P   C��xAs��   As�P   As�P   C�һAs�   As�P   AseP   C���As��   AseP   As�P   C�ͯAs��   As�P   AsGP   C��Asl�   AsGP   As�P   C���As��   As�P   As)P   C��AsN�   As)P   As�P   C���As��   As�P   AsP   C���As0�   AsP   As!|P   C��!As!��   As!|P   As#�P   C��As$�   As#�P   As&^P   C���As&��   As&^P   As(�P   C��aAs(��   As(�P   As+@P   C��ZAs+e�   As+@P   As-�P   C���As-��   As-�P   As0"P   C��)As0G�   As0"P   As2�P   C��As2��   As2�P   As5P   C��,As5)�   As5P   As7uP   C���As7��   As7uP   As9�P   C��As:�   As9�P   As<WP   C��As<|�   As<WP   As>�P   C�As>��   As>�P   AsA9P   C��AsA^�   AsA9P   AsC�P   C�	AsC��   AsC�P   AsFP   C��AsF@�   AsFP   AsH�P   C��AsH��   AsH�P   AsJ�P   C�uAsK"�   AsJ�P   AsMnP   C�WAsM��   AsMnP   AsO�P   C�6AsP�   AsO�P   AsRPP   C��AsRu�   AsRPP   AsT�P   C�5�AsT��   AsT�P   AsW2P   C�=�AsWW�   AsW2P   AsY�P   C�F6AsY��   AsY�P   As\P   C�:�As\9�   As\P   As^�P   C�9oAs^��   As^�P   As`�P   C�OpAsa�   As`�P   AscgP   C�H�Asc��   AscgP   Ase�P   C�R�Ase��   Ase�P   AshIP   C�]�Ashn�   AshIP   Asj�P   C�^�Asj��   Asj�P   Asm+P   C�jAsmP�   Asm+P   Aso�P   C�qhAso��   Aso�P   AsrP   C�lAsr2�   AsrP   Ast~P   C���Ast��   Ast~P   Asv�P   C���Asw�   Asv�P   Asy`P   C��Asy��   Asy`P   As{�P   C�~�As{��   As{�P   As~BP   C�z8As~g�   As~BP   As��P   C��1As���   As��P   As�$P   C���As�I�   As�$P   As��P   C��aAs���   As��P   As�P   C���As�+�   As�P   As�wP   C��As���   As�wP   As��P   C���As��   As��P   As�YP   C��pAs�~�   As�YP   As��P   C���As���   As��P   As�;P   C��<As�`�   As�;P   As��P   C���As���   As��P   As�P   C��As�B�   As�P   As��P   C��4As���   As��P   As��P   C�ٍAs�$�   As��P   As�pP   C���As���   As�pP   As��P   C��DAs��   As��P   As�RP   C��As�w�   As�RP   As��P   C�As���   As��P   As�4P   C�pAs�Y�   As�4P   As��P   C��As���   As��P   As�P   C�|As�;�   As�P   As��P   C� �As���   As��P   As��P   C�KAs��   As��P   As�iP   C��As���   As�iP   As��P   C�9�As���   As��P   As�KP   C�HfAs�p�   As�KP   As��P   C�"�As���   As��P   As�-P   C��As�R�   As�-P   AsP   C�9�As���   AsP   As�P   C�GZAs�4�   As�P   AsǀP   C�S�Asǥ�   AsǀP   As��P   C�\�As��   As��P   As�bP   C�W0Aṡ�   As�bP   As��P   C�WwAs���   As��P   As�DP   C�vAs�i�   As�DP   AsӵP   C�rJAs���   AsӵP   As�&P   C�|As�K�   As�&P   AsؗP   C���Asؼ�   AsؗP   As�P   C��<As�-�   As�P   As�yP   C��HAsݞ�   As�yP   As��P   C���As��   As��P   As�[P   C���As��   As�[P   As��P   C���As���   As��P   As�=P   C���As�b�   As�=P   As�P   C���As���   As�P   As�P   C��6As�D�   As�P   As�P   C��oAs��   As�P   As�P   C��As�&�   As�P   As�rP   C��aAs��   As�rP   As��P   C��WAs��   As��P   As�TP   C��As�y�   As�TP   As��P   C��2As���   As��P   As�6P   C�mAs�[�   As�6P   As��P   C�As���   As��P   AtP   C�\At=�   AtP   At�P   C��At��   At�P   At�P   C�(�At�   At�P   At	kP   C�*�