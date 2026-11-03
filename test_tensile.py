import subprocess
import os
import numpy as np

in_script = '/home/wangj/epoxy_prod/data/test_tensile.in'
log_file = '/home/wangj/epoxy_prod/data/test_tensile.log'
data_file = '/home/wangj/epoxy_prod/data/snapshot_r_1.00_alpha_0.80_relaxed.data'

script_content = f"""units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
special_bonds lj 0.0 0.0 0.5
read_data {data_file}
pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6
bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

velocity all create 300.0 42 dist gaussian
fix 1 all nvt temp 300.0 300.0 50.0
run 200
unfix 1

# 单轴微拉伸 (x方向拉伸 1.0% 应变)
fix 2 all deform 1 x scale 1.010 y volume z volume remap x
fix 3 all nvt temp 300.0 300.0 50.0
thermo 20
thermo_style custom step lx ly lz pxx pyy pzz
log {log_file}
run 200
unfix 2
unfix 3
"""

with open(in_script, 'w') as f:
    f.write(script_content)

env = os.environ.copy()
env['OMP_NUM_THREADS'] = '16'
cmd = ['/opt/conda/envs/lammps/bin/lmp', '-sf', 'omp', '-pk', 'omp', '16', '-in', in_script]
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
print('Return code:', res.returncode)
if os.path.exists(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    for line in lines[-15:]:
        print(line.strip())
