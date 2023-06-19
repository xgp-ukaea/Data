"""
Quick script for viewing and altering npz files
"""
from numpy import load, array, exp, savez
lines_lwr = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
lines_upr = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
index = 4
line_lwr = lines_lwr[index]
line_upr = lines_upr[index]
atm_fn = f'd_{line_upr}_pec_data (1).npz'
atm_data = load(atm_fn)
ln_te = atm_data['ln_te']
ln_ne = atm_data['ln_ne']
exc_ln_pec = atm_data['exc_ln_pec']
rec_ln_pec = atm_data['rec_ln_pec']
mol_ln_pec = atm_data['mol_ln_pec']
# print(exp(ln_te)[:74])
# print(exp(ln_ne)[:74])
# print(rec_ln_pec.shape)
# print(rec_ln_pec[:74])
for d in atm_data:
    print(d)
lim_index = 74
savez(f'd_{line_lwr}_pec_data',
    ln_te=ln_te,
    ln_ne=ln_ne[:lim_index],
    exc_ln_pec=exc_ln_pec[:lim_index],
    rec_ln_pec=rec_ln_pec[:lim_index],
)
savez(f'd_{line_lwr}_molecular_pec_data',
    ln_te=ln_te,
    ln_ne=ln_ne[:lim_index],
    eff_mol_ln_pec=mol_ln_pec[:lim_index],
)
# for d in mol_data:
#     print(d)
# for atm:
#     ln_te, ln_ne, exc_ln_pec, rec_ln_pec
# for mol:
#     ln_te, ln_ne, eff_mol_ln_pec
quit()
print(exp(atm_data['ln_te']))

for line in ('alpha', 'beta', 'gamma', 'delta', 'epsilon'):
    atm_fn = f'd_{line}_pec_data.npz'
    mol_fn = f'd_{line}_molecular_pec_data.npz'
    atm_data = load(atm_fn)
    mol_data = load(mol_fn)
    # compare the arrays
    for param in ('ln_te', 'ln_ne'):
        if not array(atm_data[param] == mol_data[param]).all():
            print(line, param, ':')
            print(exp(atm_data[param]))
            print(exp(mol_data[param]))
# print(mol_data['ln_te'])