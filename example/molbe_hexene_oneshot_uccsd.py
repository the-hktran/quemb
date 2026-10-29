# Illustrates a one-shot BE UCCSD calculation from UHF reference
# for hexene anion


from pyscf import gto, scf
import numpy as np
from quemb.molbe import UBE, fragmentate

# Give path to structure xyz file
structure = "data/hexene.xyz"

# Build PySCF molecule object
mol = gto.M()
mol.atom = structure
mol.basis = "sto-3g"
mol.charge = -1
mol.spin = 1
mol.build()

# Run UHF with PySCF
mf = scf.UHF(mol)
mf.kernel()

# Specify number of processors
nproc = 1

# Initialize fragments without frozen core approximation at BE2 level
fobj = fragmentate(frag_type="autogen", n_BE=2, mol=mol, frozen_core=False)
# Initialize UBE
mybe = UBE(mf, fobj)

# Perform one round of BE, without density or chemical potential matching,
# and return the energy.
# clean_eri will delete all of the ERI files from scratch
mybe.oneshot(solver="UCCSD", nproc=nproc)

# Compare density matrices from BE and CCSD
be_rdm1a, be_rdm1b = mybe.urdm1_fullbasis()
be_rdm2aa, be_rdm2ab, be_rdm2bb = mybe.urdm2_fullbasis()

from pyscf import cc
mycc = cc.UCCSD(mf)
mycc.kernel()

cc_rdm1a, cc_rdm1b = mycc.make_rdm1()
cc_rdm2aa, cc_rdm2ab, cc_rdm2bb = mycc.make_rdm2()

cc_rdm1a = mf.mo_coeff[0] @ cc_rdm1a @ mf.mo_coeff[0].T
cc_rdm1b = mf.mo_coeff[1] @ cc_rdm1b @ mf.mo_coeff[1].T
cc_rdm2aa = np.einsum("pi,qj,rk,sl,ijkl->pqrs", mf.mo_coeff[0], mf.mo_coeff[0], mf.mo_coeff[0], mf.mo_coeff[0], cc_rdm2aa, optimize=True)
cc_rdm2ab = np.einsum("pi,qj,rk,sl,ijkl->pqrs", mf.mo_coeff[0], mf.mo_coeff[0], mf.mo_coeff[1], mf.mo_coeff[1], cc_rdm2ab, optimize=True)
cc_rdm2bb = np.einsum("pi,qj,rk,sl,ijkl->pqrs", mf.mo_coeff[1], mf.mo_coeff[1], mf.mo_coeff[1], mf.mo_coeff[1], cc_rdm2bb, optimize=True)

print("Comparing BE and CCSD density matrices:")
print("1-RDM alpha difference:", np.linalg.norm(be_rdm1a - cc_rdm1a), "from", np.linalg.norm(cc_rdm1a))
print("1-RDM beta difference:", np.linalg.norm(be_rdm1b - cc_rdm1b), "from", np.linalg.norm(cc_rdm1b))
print("2-RDM alpha-alpha difference:", np.linalg.norm(be_rdm2aa - cc_rdm2aa), "from", np.linalg.norm(cc_rdm2aa))
print("2-RDM alpha-beta difference:", np.linalg.norm(be_rdm2ab - cc_rdm2ab), "from", np.linalg.norm(cc_rdm2ab))
print("2-RDM beta-beta difference:", np.linalg.norm(be_rdm2bb - cc_rdm2bb), "from", np.linalg.norm(cc_rdm2bb))
