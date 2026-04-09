"""
Test suite — updated for new binding signatures:
  - randomized_svd returns (np.ndarray, list, np.ndarray)
  - tfidf returns (MatrixCSR_double, list)
"""
import numpy as np
import sys

def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")

def check(name, passed, detail=""):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return passed

def allclose(a, b, tol=1e-6):
    return np.allclose(np.array(a,dtype=float), np.array(b,dtype=float), atol=tol)

section("Importing modules")
try:
    import linear_algebra
    import linear_algebra.dense as dense
    import linear_algebra.csr   as csr
    import decomposition        as rsvd_mod
    import tfidf                as tfidf_mod
    Mat = dense.Matrixdouble
    CSR = csr.MatrixCSR_double
    print("  [PASS] all modules imported")
    print(f"  dense:  {[x for x in dir(dense) if not x.startswith('_')]}")
    print(f"  csr:    {[x for x in dir(csr)   if not x.startswith('_')]}")
    print(f"  decomp: {[x for x in dir(rsvd_mod) if not x.startswith('_')]}")
    print(f"  tfidf:  {[x for x in dir(tfidf_mod) if not x.startswith('_')]}")
except ImportError as e:
    print(f"  [FAIL] {e}"); sys.exit(1)

# ── 1. Dense Matrix ──────────────────────────────────────
section("1. Dense Matrix")
M  = Mat(3,3);  check("zero construction", M.rows==3 and M.cols==3)
I3 = Mat(3);    check("identity", I3[0,0]==1.0 and I3[0,1]==0.0)
A  = Mat([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,10.0]])
check("from list", A[0,0]==1.0 and A[2,2]==10.0)
np_A=np.array([[1,2,3],[4,5,6],[7,8,10]],dtype=float)
check("A*I=A",    allclose((A*I3).to_numpy(), np_A))
At=A.transpose(); check("transpose dims",   At.rows==3 and At.cols==3)
check("transpose values", At[0,1]==A[1,0])
check("mat*vec",  allclose(A*[1.0,0.0,0.0], [1.0,4.0,7.0]))
check("mat_T_vec",allclose(A.mat_transpose_vec([1.0,0.0,0.0]), [1.0,2.0,3.0]))

# ── 2. Gram-Schmidt (replaces QR) ────────────────────────
section("2. Gram-Schmidt")
def test_gs(name, data):
    A  = Mat(data)
    np_D=np.array(data,dtype=float)
    Q  = dense.gram_schmidt_double(A)
    np_Q=Q.to_numpy()
    orth_err=np.linalg.norm(np_Q.T@np_Q - np.eye(np_Q.shape[1]))
    check(f"{name}: QtQ=I", orth_err<1e-8, f"err={orth_err:.2e}")
    # columns of Q span col space of A: Qᵀ(A - Q Qᵀ A) should be ~0
    proj_err=np.linalg.norm(np_D - np_Q@(np_Q.T@np_D))
    check(f"{name}: Q spans col(A)", proj_err<1e-8, f"err={proj_err:.2e}")

test_gs("4x3", [[1,2,3],[4,5,6],[7,8,10],[2,3,5]])
test_gs("5x3", [[1,2,1],[0,3,1],[2,1,4],[1,0,2],[3,2,1]])
test_gs("3x3", [[2,1,3],[1,4,2],[3,2,5]])

# ── 3. SVD Jacobi ────────────────────────────────────────
section("3. SVD Jacobi")
def test_svd(name, data):
    A=Mat(data); np_D=np.array(data,dtype=float)
    (U,V),sigma=dense.svd_jacobi(A)
    np_U=U.to_numpy(); np_V=V.to_numpy(); sig=np.array(sigma)
    check(f"{name}: sigma>=0",    np.all(sig>=0))
    check(f"{name}: decreasing",  np.all(np.diff(sig)<=1e-9))
    rec_err=np.linalg.norm(np_U@np.diag(sig)@np_V.T - np_D)
    check(f"{name}: reconstruction", rec_err<1e-8, f"err={rec_err:.2e}")
    orth_err=np.linalg.norm(np_U.T@np_U - np.eye(np_U.shape[1]))
    check(f"{name}: U orthonormal",  orth_err<1e-6, f"err={orth_err:.2e}")

test_svd("3x2",[[1,2],[3,4],[5,6]])
test_svd("4x3",[[1,2,3],[4,5,6],[7,8,10],[2,1,3]])
test_svd("3x3",[[2,1,0],[1,3,1],[0,1,2]])

# ── 4. CSR Matrix ────────────────────────────────────────
section("4. CSR Matrix")
coo=[((0,0),1.0),((0,2),2.0),((1,1),3.0),((2,0),4.0),((2,2),5.0)]
A_csr=CSR(coo,3,3)
check("construction",   A_csr.row_size==3 and A_csr.col_size==3)
check("SpMV",           allclose(A_csr*[1.0,1.0,1.0], [3.0,3.0,9.0]))
check("SpMV_transpose", allclose(A_csr.spmv_transpose([1.0,1.0,1.0]), [5.0,3.0,7.0]))
AT=A_csr.transpose()
check("Transpose dims", AT.row_size==3 and AT.col_size==3)
np_M=np.array([[1,0,2],[0,3,0],[4,0,5]],dtype=float)
D=Mat([[1.0,0.0],[0.0,1.0],[1.0,1.0]])
check("CSR*Dense", allclose((A_csr*D).to_numpy(), np_M@np.array([[1,0],[0,1],[1,1]],dtype=float)))
check("to_numpy",  allclose(A_csr.to_numpy(), np_M))

# ── 5. TF-IDF ────────────────────────────────────────────
section("5. TF-IDF")
docs=[[0,1,1,2],[2,3,3],[0,4,4,4]]; vocab_size=5
try:
    tfidf_mat,idf=tfidf_mod.tfidf_double(docs,vocab_size)
    check("dims",        tfidf_mat.row_size==vocab_size and tfidf_mat.col_size==3)
    check("idf length",  len(idf)==vocab_size)
    check("idf>=1",      all(v>=1.0 for v in idf))
    check("rarer=higher idf", idf[3]>idf[2])
    check("values positive",  all(v>=0 for v in tfidf_mat*[1.0]*3))
except Exception as e: print(f"  [FAIL] {e}")

# ── 6. Randomized SVD ────────────────────────────────────
section("6. Randomized SVD")
def make_csr(m,n,rank,seed=42):
    np.random.seed(seed)
    A=np.random.randn(m,rank)@np.random.randn(rank,n)+0.01*np.random.randn(m,n)
    A[np.abs(A)<0.3]=0.0
    coo=[((i,j),float(A[i,j])) for i in range(m) for j in range(n) if A[i,j]!=0.0]
    return CSR(coo,m,n), A

m,n,r=30,20,4; k,p,q=4,4,2
A_csr,A_dense=make_csr(m,n,r)
try:
    U,sigma,V=rsvd_mod.randomized_svd(A_csr,k=k,p=p,q=q)
    # U and V are numpy arrays directly
    sig=np.array(sigma)
    check("U dims",          U.shape==(m,k))
    check("V dims",          V.shape==(n,k))
    check("sigma length",    len(sig)==k)
    check("sigma positive",  np.all(sig>0))
    check("sigma decreasing",np.all(np.diff(sig)<=1e-6))
    orth_err=np.linalg.norm(U.T@U - np.eye(k))
    check("U orthonormal",   orth_err<1e-6, f"err={orth_err:.2e}")
    rel=np.linalg.norm(U@np.diag(sig)@V.T - A_dense)/np.linalg.norm(A_dense)
    check("reconstruction<30%", rel<0.30, f"err={rel:.2%}")
except Exception as e: print(f"  [FAIL] {e}")

# ── 7. End-to-end ────────────────────────────────────────
section("7. End-to-end: TF-IDF → RSVD → fold-in")
try:
    np.random.seed(0)
    docs_e2e=[list(np.random.randint(0,50,size=np.random.randint(5,15))) for _ in range(20)]
    tfidf_mat,idf=tfidf_mod.tfidf_double(docs_e2e,50)
    U,sigma,V=rsvd_mod.randomized_svd(tfidf_mat,k=4,p=4,q=1)
    sig=np.array(sigma); idf_a=np.array(idf)
    check("U shape", U.shape==(50,4))
    q_tf=np.zeros(50); q_tf[3]=1.0; q_tf[7]=1.0
    q_embed=(U/sig).T @ (q_tf*idf_a)
    check("query embed shape",  q_embed.shape==(4,))
    check("query embed finite", np.all(np.isfinite(q_embed)))
    check("pipeline complete",  True)
except Exception as e: print(f"  [FAIL] {e}")

print(f"\n{'='*55}\n  All tests complete.\n{'='*55}\n")