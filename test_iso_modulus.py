import os
import subprocess

def run_modulus_test(data_file):
    # LAMMPS 模板，应变 0.002
    lmp_script = f"""units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data {data_file}
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

minimize 1.0e-5 1.0e-7 500 1500
run 0
write_data /tmp/tmp_base.data
clear

# ================= Z 轴测试 =================
units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_base.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale 0.99924 y scale 0.99924 z scale 1.0020 remap
minimize 1.0e-5 1.0e-7 400 1200
run 0
variable pzz_val equal pzz
variable pxx_val equal pxx
variable pyy_val equal pyy
print "Z_POS: PZZ=$(pzz) PXX=$(pxx) PYY=$(pyy)"
clear

units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_base.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale 1.00076 y scale 1.00076 z scale 0.9980 remap
minimize 1.0e-5 1.0e-7 400 1200
run 0
print "Z_NEG: PZZ=$(pzz) PXX=$(pxx) PYY=$(pyy)"
clear

# ================= X 轴测试 =================
units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_base.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale 1.0020 y scale 0.99924 z scale 0.99924 remap
minimize 1.0e-5 1.0e-7 400 1200
run 0
print "X_POS: PZZ=$(pzz) PXX=$(pxx) PYY=$(pyy)"
clear

units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_base.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale 0.9980 y scale 1.00076 z scale 1.00076 remap
minimize 1.0e-5 1.0e-7 400 1200
run 0
print "X_NEG: PZZ=$(pzz) PXX=$(pxx) PYY=$(pyy)"
clear

# ================= Y 轴测试 =================
units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_base.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale 0.99924 y scale 1.0020 z scale 0.99924 remap
minimize 1.0e-5 1.0e-7 400 1200
run 0
print "Y_POS: PZZ=$(pzz) PXX=$(pxx) PYY=$(pyy)"
clear

units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_base.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale 1.00076 y scale 0.9980 z scale 1.00076 remap
minimize 1.0e-5 1.0e-7 400 1200
run 0
print "Y_NEG: PZZ=$(pzz) PXX=$(pxx) PYY=$(pyy)"
clear
"""
    with open('/tmp/test_relax.in', 'w') as f:
        f.write(lmp_script)
        
    cmd = ['/opt/conda/envs/lammps/bin/lmp', '-sf', 'omp', '-pk', 'omp', '16', '-in', '/tmp/test_relax.in']
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    vals = {}
    for line in res.stdout.splitlines():
        for prefix in ['Z_POS:', 'Z_NEG:', 'X_POS:', 'X_NEG:', 'Y_POS:', 'Y_NEG:']:
            if prefix in line:
                vals[prefix] = {}
                for item in line.split()[1:]:
                    k, v = item.split('=')
                    vals[prefix][k] = float(v)
                    
    BAR_TO_GPA = 0.000101325
    STRAIN_SPAN = 0.004 # +0.002 - (-0.002)
    
    # 轴向差分等效应力
    sig_z_pos = -vals['Z_POS:']['PZZ'] * BAR_TO_GPA
    sig_zt_pos = -0.5 * (vals['Z_POS:']['PXX'] + vals['Z_POS:']['PYY']) * BAR_TO_GPA
    eq_z_pos = sig_z_pos - sig_zt_pos
    
    sig_z_neg = -vals['Z_NEG:']['PZZ'] * BAR_TO_GPA
    sig_zt_neg = -0.5 * (vals['Z_NEG:']['PXX'] + vals['Z_NEG:']['PYY']) * BAR_TO_GPA
    eq_z_neg = sig_z_neg - sig_zt_neg
    Ez = (eq_z_pos - eq_z_neg) / STRAIN_SPAN
    
    # X-axis
    sig_x_pos = -vals['X_POS:']['PXX'] * BAR_TO_GPA
    sig_xt_pos = -0.5 * (vals['X_POS:']['PYY'] + vals['X_POS:']['PZZ']) * BAR_TO_GPA
    eq_x_pos = sig_x_pos - sig_xt_pos
    
    sig_x_neg = -vals['X_NEG:']['PXX'] * BAR_TO_GPA
    sig_xt_neg = -0.5 * (vals['X_NEG:']['PYY'] + vals['X_NEG:']['PZZ']) * BAR_TO_GPA
    eq_x_neg = sig_x_neg - sig_xt_neg
    Ex = (eq_x_pos - eq_x_neg) / STRAIN_SPAN
    
    # Y-axis
    sig_y_pos = -vals['Y_POS:']['PYY'] * BAR_TO_GPA
    sig_yt_pos = -0.5 * (vals['Y_POS:']['PXX'] + vals['Y_POS:']['PZZ']) * BAR_TO_GPA
    eq_y_pos = sig_y_pos - sig_yt_pos
    
    sig_y_neg = -vals['Y_NEG:']['PYY'] * BAR_TO_GPA
    sig_yt_neg = -0.5 * (vals['Y_NEG:']['PYY'] + vals['Y_NEG:']['PZZ']) * BAR_TO_GPA
    eq_y_neg = sig_y_neg - sig_yt_neg
    Ey = (eq_y_pos - eq_y_neg) / STRAIN_SPAN
    
    E_iso = (Ex + Ey + Ez) / 3.0
    return Ex, Ey, Ez, E_iso

if __name__ == '__main__':
    targets = [
        (1.0, 0.7),
        (1.0, 0.8),
        (1.0, 0.9),
        (0.8, 0.9),
        (1.2, 0.9)
    ]
    for r, a in targets:
        fpath = f'/home/wangj/epoxy_prod/data/prop_r_{r:.2f}_alpha_{a:.2f}_300K.data'
        Ex, Ey, Ez, E_iso = run_modulus_test(fpath)
        print(f"r={r:.2f}, alpha={a:.2f} -> Ex={Ex:.4f}, Ey={Ey:.4f}, Ez={Ez:.4f} | E_iso={E_iso:.4f} GPa")
