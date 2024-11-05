#!/bin/bash

# seq FIRST STEP LAST

#python3 ~/csbosonscpp/tools/stats.py -f operators0.dat -o ReMz ImMz ReMz_squared ImMz_squared ReNup ImNup ReNdn ImNdn -a -q > data.dat

#python3 ~/csbosonscpp/tools/stats.py -f operators0.dat -o ReN ImN ReN_squared ImN_squared -a -q > data0.dat
python3 ~/CSBosonsCpp/tools/stats.py -f operators0.dat -o Repartnum0 Impartnum0 Repartnum1 Impartnum1 Repartnum0_squared Impartnum0_squared Repartnum1_squared Impartnum1_squared Repartnum Impartnum Repartnum_squared Impartnum_squared ReNK0_0 ImNK0_0 ReNK0_1 ImNK0_1 ReNK0_tot ImNK0_tot Retot_beta_Pressure_Int Imtot_beta_Pressure_Int ReBeta_Stress_x ImBeta_Stress_x ReBeta_Stress_y ImBeta_Stress_y ReE ImE ReE_squared ImE_squared ReMag_x ImMag_x ReMag_y ImMag_y ReMag_z ImMag_z ReMag_x_squared ImMag_x_squared ReMag_y_squared ImMag_y_squared ReMag_z_squared ImMag_z_squared ReN_ring ImN_ring ReNF_density_xx ImNF_density_xx ReNF_density_xy ImNF_density_xy ReNF_density_yy ImNF_density_yy RePx_phys ImPx_phys RePy_phys ImPy_phys -a -q > data0.dat


