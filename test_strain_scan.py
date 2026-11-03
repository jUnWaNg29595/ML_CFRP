import subprocess

def test_strain_range(data_file):
    print(f"\n--- Testing {data_file} ---")
    # 测试不同的应变幅值: 0.002, 0.004, 0.006, 0.008
    for eps in [0.002, 0.003, 0.004, 0.005, 0.006, 0.008]:
        scale_pos_z = 1.0 + eps
        scale_pos_t = 1.0 - 0.38 * eps
        scale_neg_z = 1.0 - eps
        scale_neg_t = 1.0 + 0.38 * eps
        
        script = f"""units real
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

minimize 1.0e-5 1.0e-7 200 600
run 0
write_data /tmp/tmp_scan.data
clear

units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_scan.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale {scale_pos_t:.6f} y scale {scale_pos_t:.6f} z scale {scale_pos_z:.6f} remap
minimize 1.0e-5 1.0e-7 200 600
run 0
variable pzz_pos equal pzz
variable pxx_pos equal pxx
variable pyy_pos equal pyy
print "POS: PZZ=${{pzz_pos}} PXX=${{pxx_pos}} PYY=${{pyy_pos}}"
clear

units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_scan.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale {scale_neg_t:.6f} y scale {scale_neg_t:.6f} z scale {scale_neg_z:.6f} remap
minimize 1.0e-5 1.0e-7 200 600
run 0
variable pzz_neg equal pzz
variable pxx_neg equal pxx
variable pyy_neg equal pyy
print "NEG: PZZ=${{pzz_neg}} PXX=${{pxx_neg}} PYY=${{pyy_neg}}"
"""
        with open('/tmp/scan.in', 'w') as f:
            f.write(script)
        cmd = ['/opt/conda/envs/lammps/bin/lmp', '-sf', 'omp', '-pk', 'omp', '16', '-in', '/tmp/scan.in']
        out = subprocess.check_output(cmd, text=True)
        
        pos_vals, neg_vals = {}, {}
        for line in out.splitlines():
            if 'POS: PZZ=' in line:
                for item in line.split()[1:]:
                    k, v = item.split('=')
                    pos_vals[k] = float(v)
            elif 'NEG: PZZ=' in line:
                for item in line.split()[1:]:
                    k, v = item.split('=')
                    neg_vals[k] = float(v)
                    
        BAR_TO_GPA = 0.000101325
        sig_z_pos = -pos_vals['PZZ'] * BAR_TO_GPA
        sig_t_pos = -0.5 * (pos_vals['PXX'] + pos_vals['PYY']) * BAR_TO_GPA
        eq_pos = sig_z_pos - sig_t_pos
        
        sig_z_neg = -neg_vals['PZZ'] * BAR_TO_GPA
        sig_t_neg = -0.5 * (neg_vals['PXX'] + neg_vals['PYY']) * BAR_TO_GPA
        eq_neg = sig_z_neg - sig_t_neg
        
        E = (eq_pos - eq_neg) / (2 * eps)
        print(f"eps={eps:.4f} -> PZZ_pos={pos_vals['PZZ']:.1f}, PZZ_neg={neg_vals['PZZ']:.1f} | E={E:.4f} GPa")

if __name__ == '__main__':
    for r, a in [(1.0, 0.7), (1.0, 0.9), (1.2, 0.9)]:
        fpath = f'/home/wangj/epoxy_prod/data/prop_r_{r:.2f}_alpha_{a:.2f}_300K.data'
        test_strain_range(fpath)
