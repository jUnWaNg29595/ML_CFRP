import subprocess

for fname, r, alpha in [
    ('prop_r_0.80_alpha_0.70_300K.data', 0.80, 0.70),
    ('prop_r_1.20_alpha_0.90_300K.data', 1.20, 0.90),
]:
    data = f'/home/wangj/epoxy_prod/data/{fname}'
    script = f"""units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data {data}
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
write_data /tmp/tmp_d.data
clear

units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_d.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale 0.99696 y scale 0.99696 z scale 1.0080 remap
minimize 1.0e-5 1.0e-7 200 600
run 0
variable pzz_pos equal pzz
variable pxx_pos equal pxx
variable pyy_pos equal pyy
variable pe_pos equal pe
print "VAL_POS: PZZ=${{pzz_pos}} PXX=${{pxx_pos}} PYY=${{pyy_pos}} PE=${{pe_pos}}"
clear

units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data /tmp/tmp_d.data
special_bonds lj 0.0 0.0 0.5
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

change_box all x scale 1.00304 y scale 1.00304 z scale 0.9920 remap
minimize 1.0e-5 1.0e-7 200 600
run 0
variable pzz_neg equal pzz
variable pxx_neg equal pxx
variable pyy_neg equal pyy
variable pe_neg equal pe
print "VAL_NEG: PZZ=${{pzz_neg}} PXX=${{pxx_neg}} PYY=${{pyy_neg}} PE=${{pe_neg}}"
"""
    with open('/tmp/d_run.in', 'w') as f:
        f.write(script)
    cmd = ['/opt/conda/envs/lammps/bin/lmp', '-sf', 'omp', '-pk', 'omp', '16', '-in', '/tmp/d_run.in']
    out = subprocess.check_output(cmd, text=True)
    print(f'=== {fname} ===')
    for line in out.splitlines():
        if 'VAL_POS:' in line or 'VAL_NEG:' in line:
            print(line)
