# UBE Calculation of Ethane anion using UCCSD and comparison to CCSD gradients
from pyscf import gto, scf, cc, grad
import numpy as np
from quemb.molbe import UBE, fragmentate

ethane_atoms ="""
C 0.000000 0.000000 0.000000
C 1.540000 0.000000 0.000000
H -0.540000 0.900000 0.000000
H -0.540000 -0.900000 0.000000
H 2.080000 0.900000 0.000000
H 2.080000 -0.900000 0.000000
"""

hexene_file = "data/hexene.xyz"

# Build PySCF molecule object
mol = gto.M()
mol.atom = hexene_file
mol.basis = "sto-3g"
mol.charge = -1
mol.spin = 1
mol.build()

# Run UHF with PySCF
mf = scf.UHF(mol)
mf.kernel()

# Specify number of processors
nproc = 1

# Initialize fragments without frozen core approximation at BE1 level
fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, frozen_core=False)
# Initialize UBE
mybe = UBE(mf, fobj)

# Perform one round of BE, without density or chemical potential matching,
mybe.oneshot(solver="UCCSD", nproc=nproc)

rdm1 = mybe.urdm1_fullbasis()
rdm2 = mybe.urdm2_fullbasis()

mcc = cc.CCSD(mf, frozen=False)
mcc.kernel()

true_rdm1_mo = mcc.make_rdm1()
true_rdm2_mo = mcc.make_rdm2()

C = mf.mo_coeff

true_rdm1a = np.einsum("pi,ij,qj->pq", C[0], true_rdm1_mo[0], C[0])
true_rdm1b = np.einsum("pi,ij,qj->pq", C[1], true_rdm1_mo[1], C[1])
true_rdm2aa = np.einsum("ijkl,pi,qj,rk,sl->pqrs", true_rdm2_mo[0], C[0], C[0], C[0], C[0], optimize=True)
true_rdm2ab = np.einsum("ijkl,pi,qj,rk,sl->pqrs", true_rdm2_mo[1], C[0], C[0], C[1], C[1], optimize=True)
true_rdm2bb = np.einsum("ijkl,pi,qj,rk,sl->pqrs", true_rdm2_mo[2], C[1], C[1], C[1], C[1], optimize=True)

print("Comparison of RDMs")
print("Norm of difference between BE and CCSD RDM1s versus norm of CCSD RDM1s")
print(np.linalg.norm(rdm1[0] - true_rdm1a), np.linalg.norm(true_rdm1a))
print(np.linalg.norm(rdm1[1] - true_rdm1b), np.linalg.norm(true_rdm1b))

print("Norm of difference between BE and CCSD RDM2s versus norm of CCSD RDM2s")
print(np.linalg.norm(rdm2[0] - true_rdm2aa), np.linalg.norm(true_rdm2aa))
print(np.linalg.norm(rdm2[1] - true_rdm2ab), np.linalg.norm(true_rdm2ab))
print(np.linalg.norm(rdm2[2] - true_rdm2bb), np.linalg.norm(true_rdm2bb))


# Compare gradients
mygrad_cc = mcc.Gradients()
grad_cc = mygrad_cc.kernel()
grad_cc = mcc.nuc_grad_method().kernel()

grad_be = mybe.get_be_grad()

print("Comparison of Gradients")
print("CCSD Gradient")
print(grad_cc)
print("BE Gradient")
print(grad_be)

print("Gradient Results")
print("Norm of CCSD Gradient: ", np.linalg.norm(grad_cc))
print("Norm of BE Gradient: ", np.linalg.norm(grad_be))
print("Norm of Difference: ", np.linalg.norm(grad_cc - grad_be))
