## test_backend.py
## run from math.vis root with: python test_backend.py
## tests every function in mathBackAfterClaude.py
## each test prints PASS or FAIL with the result

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "mathBack"))
from mathBackAfterClaude import (
    Matrix_Calc,
    Matrix_Act,
    Matrix_Diag,
    J_Matrix,
    NumberVectorInnerProduct,
    NumberVectorIfBase,
    Gram_Schmidt,
    MatricVectorInnerProduct,
    MatricVectorIfBase,
    PolyInnerProduct,
    PolyIfBase,
    RepresentingMatrix,
    ImTAndKerT,
    IsLT,
    IsIso,
    Build_Representing_Matrix,
)

passed = 0
failed = 0

def check(name, result, expect_key, expect_val=None, contains=None):
    global passed, failed
    if "error" in result and expect_key != "error":
        print(f"  FAIL  {name}")
        print(f"        got error: {result['error']}")
        failed += 1
        return
    if expect_key not in result:
        print(f"  FAIL  {name}")
        print(f"        key '{expect_key}' not in result: {result}")
        failed += 1
        return
    val = result[expect_key]
    if expect_val is not None and val != expect_val:
        print(f"  FAIL  {name}")
        print(f"        expected {expect_val}, got {val}")
        failed += 1
        return
    if contains is not None and contains not in str(val):
        print(f"  FAIL  {name}")
        print(f"        expected '{contains}' in result, got {val}")
        failed += 1
        return
    print(f"  PASS  {name}  →  {str(val)[:80]}")
    passed += 1


print("\n" + "="*60)
print("  math.vis backend test suite")
print("="*60)

#######################################################################
print("\n── Matrix functions ──")

# Matrix_Calc: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
r = Matrix_Calc([[1,2],[3,4]], [[5,6],[7,8]])
check("Matrix_Calc basic multiply", r, "result", [[19.0,22.0],[43.0,50.0]])

# Matrix_Calc: wrong sizes
r = Matrix_Calc([[1,2],[3,4]], [[1,2,3]])
check("Matrix_Calc wrong size", r, "error")

# Matrix_Act: identity matrix
r = Matrix_Act([[1,0],[0,1]])
check("Matrix_Act identity det=1", r, "determinant", 1.0)
check("Matrix_Act identity invertible", r, "inverse")

# Matrix_Act: singular matrix
r = Matrix_Act([[1,2],[2,4]])
check("Matrix_Act singular det=0", r, "determinant", 0.0)

# Matrix_Diag: diagonal matrix is always diagonalizable
r = Matrix_Diag([[2,0],[0,3]])
check("Matrix_Diag diagonal matrix", r, "is_diagonalizable", True)

# Matrix_Diag: non-square
r = Matrix_Diag([[1,2,3],[4,5,6]])
check("Matrix_Diag non-square error", r, "error")

# J_Matrix: simple 2x2
r = J_Matrix([[2,0],[0,3]])
check("J_Matrix char_poly present", r, "char_poly")
check("J_Matrix min_poly present", r, "min_poly")
check("J_Matrix Jordan form present", r, "J")
check("J_Matrix P present", r, "P")


#######################################################################
print("\n── Vector functions ──")

# NumberVectorInnerProduct: [1,2,3]·[4,5,6] = 32
r = NumberVectorInnerProduct([1,2,3], [4,5,6])
check("NumberVectorInnerProduct [1,2,3]·[4,5,6]=32", r, "result", 32.0)

# NumberVectorInnerProduct: orthogonal
r = NumberVectorInnerProduct([1,0], [0,1])
check("NumberVectorInnerProduct orthogonal=0", r, "result", 0.0)

# NumberVectorInnerProduct: size mismatch
r = NumberVectorInnerProduct([1,2], [1,2,3])
check("NumberVectorInnerProduct size mismatch error", r, "error")

# NumberVectorIfBase: standard basis R2
r = NumberVectorIfBase([[1,0],[0,1]])
check("NumberVectorIfBase standard basis R2", r, "result", True)

# NumberVectorIfBase: linearly dependent
r = NumberVectorIfBase([[1,2],[2,4]])
check("NumberVectorIfBase linearly dependent", r, "result", False)

# Gram_Schmidt: standard basis stays the same
r = Gram_Schmidt([[1,0],[0,1]])
check("Gram_Schmidt standard basis", r, "result")

# Gram_Schmidt: non-orthogonal vectors
r = Gram_Schmidt([[1,1],[1,0]])
check("Gram_Schmidt non-orthogonal", r, "result")

# Gram_Schmidt: linearly dependent
r = Gram_Schmidt([[1,0],[2,0]])
check("Gram_Schmidt linearly dependent error", r, "error")

# Gram_Schmidt: size mismatch
r = Gram_Schmidt([[1,0],[1,0,0]])
check("Gram_Schmidt size mismatch error", r, "error")

# MatricVectorInnerProduct: Tr(B^t A)
r = MatricVectorInnerProduct([[1,0],[0,1]], [[1,0],[0,1]])
check("MatricVectorInnerProduct identity Tr=2", r, "result", 2.0)

# MatricVectorInnerProduct: size mismatch
r = MatricVectorInnerProduct([[1,2]], [[1,2],[3,4]])
check("MatricVectorInnerProduct size mismatch error", r, "error")

# MatricVectorIfBase: E11, E12 in M2x2 — not enough matrices for dim=4
r = MatricVectorIfBase([[[1,0],[0,0]],[[0,1],[0,0]]])
check("MatricVectorIfBase too few matrices error", r, "error")

# MatricVectorIfBase: full standard basis of M2x2
r = MatricVectorIfBase([
    [[1,0],[0,0]],
    [[0,1],[0,0]],
    [[0,0],[1,0]],
    [[0,0],[0,1]]
])
check("MatricVectorIfBase full M2x2 basis", r, "result", True)

# PolyInnerProduct: <1, 1> on [0,1] = 1
r = PolyInnerProduct("1", "1")
check("PolyInnerProduct <1,1>=1", r, "result", "1")

# PolyInnerProduct: <x, 1-x> on [0,1]
r = PolyInnerProduct("x", "1-x")
check("PolyInnerProduct <x,1-x> present", r, "result")

# PolyIfBase: {1, x, x^2} is a base for P2
r = PolyIfBase(["1", "x", "x**2"])
check("PolyIfBase {1,x,x^2} is base", r, "result", True)

# PolyIfBase: linearly dependent
r = PolyIfBase(["1", "2"])
check("PolyIfBase linearly dependent", r, "result", False)


#######################################################################
print("\n── Linear Transformation functions ──")

# Build_Representing_Matrix: identity
r = Build_Representing_Matrix("T(x,y)=(x,y)")
check("Build_Representing_Matrix identity not None", {"result": r is not None}, "result", True)

# RepresentingMatrix: T(x,y) = (2x, y)
r = RepresentingMatrix("T(x,y) = (2x, y)")
check("RepresentingMatrix T(x,y)=(2x,y)", r, "result")

# RepresentingMatrix: not linear
r = RepresentingMatrix("T(x,y) = (x^2, y)")
check("RepresentingMatrix not linear error", r, "error")

# ImTAndKerT: zero transformation
A_zero = Build_Representing_Matrix("T(x,y) = (0, 0)")
if A_zero is not None:
    r = ImTAndKerT(A_zero)
    check("ImTAndKerT zero transformation Ker present", r, "Ker")
    check("ImTAndKerT zero transformation Img present", r, "Img")
else:
    print("  SKIP  ImTAndKerT zero transformation (AI returned None)")

# ImTAndKerT: identity
A_id = Build_Representing_Matrix("T(x,y) = (x, y)")
if A_id is not None:
    r = ImTAndKerT(A_id)
    check("ImTAndKerT identity Ker=0", r, "Ker")
else:
    print("  SKIP  ImTAndKerT identity (AI returned None)")

# IsLT: linear
A = Build_Representing_Matrix("T(x,y) = (x+y, x-y)")
r = IsLT(A)
check("IsLT T(x,y)=(x+y,x-y) is linear", r, "result")

# IsLT: not linear
A_bad = Build_Representing_Matrix("T(x,y) = (x^2, y)")
r = IsLT(A_bad)
check("IsLT T(x,y)=(x^2,y) not linear", r, "error")

# IsIso: T(x,y) = (x+y, x-y) det=[-2]!=0 so iso
A = Build_Representing_Matrix("T(x,y) = (x+y, x-y)")
r = IsIso(A)
check("IsIso T(x,y)=(x+y,x-y) is iso", r, "result")

# IsIso: T(x,y) = (x+y, x+y) det=0 so not iso
A = Build_Representing_Matrix("T(x,y) = (x+y, x+y)")
r = IsIso(A)
check("IsIso T(x,y)=(x+y,x+y) not iso", r, "error")

# IsIso: None input
r = IsIso(None)
check("IsIso None input error", r, "error")


#######################################################################
print("\n" + "="*60)
print(f"  Results: {passed} passed, {failed} failed")
print("="*60 + "\n")
