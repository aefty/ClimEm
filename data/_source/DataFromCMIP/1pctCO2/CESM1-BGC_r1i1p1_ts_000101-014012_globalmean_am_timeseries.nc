CDF   �   
      time       bnds      lon       lat          &   CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      pMon Mar 01 11:21:30 2021: cdo -s -a yearmean -fldmean /echam/folini/cmip5/1pctCO2//CESM1-BGC_r1i1p1_ts_000101-014012_fulldata.nc /echam/folini/cmip5/1pctCO2//CESM1-BGC_r1i1p1_ts_000101-014012_globalmean_am_timeseries.nc
Mon Mar 01 11:21:20 2021: cdo -s -a selvar,ts /echam/folini/cmip5/1pctCO2//tmp_01.nc /echam/folini/cmip5/1pctCO2//tmp_11.nc
Mon Mar 01 11:21:13 2021: cdo -s -a mergetime /net/atmos/data/cmip5/1pctCO2/Amon/ts/CESM1-BGC/r1i1p1/ts_Amon_CESM1-BGC_1pctCO2_r1i1p1_000101-014012.nc /echam/folini/cmip5/1pctCO2//tmp_01.nc
2012-05-12T15:01:32Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.   source        	CESM1-BGC      institution       HNSF/DOE NCAR (National Center for Atmospheric Research) Boulder, CO, USA   institute_id      NSF-DOE-NCAR   experiment_id         1pctCO2    model_id      	CESM1-BGC      forcing       GHG (CO2 only)     parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       ?�         contact       cesm_data@ucar.edu     comment       (CESM home page: http://www.cesm.ucar.edu   
references        3TBD
 See also http://www.cesm.ucar.edu/publications    initialization_method               physics_version             tracking_id       $fc73fd0d-b6fc-4b5b-9c98-083d87411962   acknowledgements     �The CESM project is supported by the National Science Foundation and the Office of Science (BER) of the U.S. Department of Energy. NCAR is sponsored by the National Science Foundation. Computing resources were provided by the Climate Simulation Laboratory at the NCAR Computational and Information Systems Laboratory (CISL), sponsored by the National Science Foundation and other agencies.      cesm_casename         b40.1850_ramp.1deg.ncbdrd.001      cesm_repotag      unknown    cesm_compset      unknown    
resolution        f09_g16 (0.9x1.25_gx1v6)   forcing_note      �Additional information on the external forcings used in this experiment can be found at http://www.cesm.ucar.edu/CMIP5/forcing_information     processed_by      8strandwg on silver.cgd.ucar.edu at 20120512  -090123.450   processing_code_information       �Last Changed Rev: 758 Last Changed Date: 2012-05-11 11:02:39 -0600 (Fri, 11 May 2012) Repository UUID: d2181dbe-5796-6825-dc7f-cbd98591f93d    product       output     
experiment        1 percent per year CO2     	frequency         year   creation_date         2012-05-12T15:01:32Z   
project_id        CMIP5      table_id      =Table Amon (12 January 2012) 4996d487f7a65749098d9cc0dccb4f8d      title         @CESM1-BGC model output prepared for CMIP5 1 percent per year CO2   parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.8.1      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               	time_bnds                                lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   ts                        standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       STS no change, CMIP5_table_comment: ""skin"" temperature (i.e., SST for open ocean)     original_name         TS     cell_methods      time: mean (interval: 30 days)     cell_measures         area: areacella    history       �2012-05-12T15:01:23Z altered by CMOR: Reordered dimensions, original order: lat lon time. 2012-05-12T15:01:23Z altered by CMOR: replaced missing value flag (-1e+32) with standard missing value (1e+20).      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CESM1-BGC_1pctCO2_r0i0p0.nc areacella: areacella_fx_CESM1-BGC_1pctCO2_r0i0p0.nc                          @��    @ú�    @ӡ@    C��k@�7P    @ӡ@    @�e@    C��]@��P    @�e@    @㔠    C���@�ߨ    @㔠    @�v�    C���@���    @�v�    @�X�    C���@���    @�X�    @�P    C���@�B�    @�P    @�P    C���@��    @�P    @��P    C��@�$�    @��P    @�pP    C��N@���    @�pP    @��P    C��U@��    @��P    @�RP    C���@�w�    @�RP    @��P    C��@���    @��P    A(    C���A,�    A(    AR�    C���Aej    AR�    A�(    C��BA��    A�(    Aè    C�ïA�j    Aè    A�(    C�©A�    A�(    A4�    C��iAGj    A4�    Am(    C�тA�    Am(    A	��    C���A	�j    A	��    A
�(    C��RA
��    A
�(    A�    C���A)j    A�    AO(    C���Aa�    AO(    A��    C��:A�j    A��    A�(    C��A��    A�(    A|T    C��A��    A|T    A�    C��A!�    A�    A��    C��	A�5    A��    AQ    C��ZAZu    AQ    A�T    C���A��    A�T    A��    C��+A��    A��    A%�    C���A/5    A%�    A�    C��.A�u    A�    A^T    C�	�Ag�    A^T    A��    C�rA�    A��    A��    C��A�5    A��    A3    C�MA<u    A3    A�T    C��Aص    A�T    Ak�    C��At�    Ak�    A�    C��A5    A�    A�    C��9A�u    A�    A@T    C�	AI�    A@T    Aܔ    C��A��    Aܔ    Ax�    C�"=A�5    Ax�    A    C�%�Au    A    A�T    C�%�A��    A�T    AM�    C�XAV�    AM�    A��    C�rA�5    A��    A�    C�%�A�u    A�    A"T    C�7A+�    A"T    A��    C�<�A��    A��    A -j    C�B)A 2�   A -j    A {�    C�B�A �:�   A {�    A ɪ    C�9�A �Z�   A ɪ    A!�    C�5fA!z�   A!�    A!e�    C�B�A!j��   A!e�    A!�
    C�QBA!���   A!�
    A"*    C�K+A"ڀ   A"*    A"PJ    C�C�A"T��   A"PJ    A"�j    C�L�A"��   A"�j    A"�    C�[�A"�:�   A"�    A#:�    C�j:A#?Z�   A#:�    A#��    C�msA#�z�   A#��    A#��    C�i�A#ۚ�   A#��    A$%
    C�N�A$)��   A$%
    A$s*    C�U�A$wڀ   A$s*    A$�J    C�q-A$���   A$�J    A%j    C��bA%�   A%j    A%]�    C��GA%b:�   A%]�    A%��    C�y�A%�Z�   A%��    A%��    C�hQA%�z�   A%��    A&G�    C�|�A&L��   A&G�    A&�
    C���A&���   A&�
    A&�*    C�oCA&�ڀ   A&�*    A'2J    C���A'6��   A'2J    A'�j    C��rA'��   A'�j    A'Ί    C��&A'�:�   A'Ί    A(�    C���A(!Z�   A(�    A(j�    C��fA(oz�   A(j�    A(��    C��@A(���   A(��    A)
    C��A)��   A)
    A)U*    C���A)Yڀ   A)U*    A)�J    C��\A)���   A)�J    A)�j    C���A)��   A)�j    A*?�    C���A*D:�   A*?�    A*��    C���A*�Z�   A*��    A*��    C��bA*�z�   A*��    A+)�    C��)A+.��   A+)�    A+x
    C���A+|��   A+x
    A+�*    C���A+�ڀ   A+�*    A,J    C��SA,��   A,J    A,bj    C���A,g�   A,bj    A,��    C��&A,�:�   A,��    A,��    C���A-Z�   A,��    A-L�    C��$A-Qz�   A-L�    A-��    C��A-���   A-��    A-�
    C���A-���   A-�
    A.7*    C��eA.;ڀ   A.7*    A.�J    C���A.���   A.�J    A.�j    C��A.��   A.�j    A/!�    C��TA/&:�   A/!�    A/o�    C�BA/tZ�   A/o�    A/��    C��A/�z�   A/��    A0�    C��A0M@   A0�    A0-    C�A0/]@   A0-    A0T    C��A0Vm@   A0T    A0{%    C�8A0}}@   A0{%    A0�5    C�$�A0��@   A0�5    A0�E    C�'fA0˝@   A0�E    A0�U    C�3�A0�@   A0�U    A1e    C�7{A1�@   A1e    A1>u    C�*A1@�@   A1>u    A1e�    C��A1g�@   A1e�    A1��    C�!�A1��@   A1��    A1��    C�4�A1��@   A1��    A1ڵ    C�9A1�@   A1ڵ    A2�    C�<�A2@   A2�    A2(�    C�@�A2+-@   A2(�    A2O�    C�OA2R=@   A2O�    A2v�    C�U>A2yM@   A2v�    A2�    C�\OA2�]@   A2�    A2�    C�XSA2�m@   A2�    A2�%    C�GBA2�}@   A2�%    A35    C�QYA3�@   A35    A3:E    C�o�A3<�@   A3:E    A3aU    C�d�A3c�@   A3aU    A3�e    C�j�A3��@   A3�e    A3�u    C�lCA3��@   A3�u    A3օ    C�x�A3��@   A3օ    A3��    C�x�A3��@   A3��    A4$�    C���A4&�@   A4$�    A4K�    C���A4N@   A4K�    A4r�    C��3A4u@   A4r�    A4��    C�{-A4�-@   A4��    A4��    C��A4�=@   A4��    A4��    C��hA4�M@   A4��    A5    C��uA5]@   A5    A56    C���A58m@   A56    A5]%    C��4A5_}@   A5]%    A5�5    C��