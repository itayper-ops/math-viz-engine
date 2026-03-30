## test_terminal.py
## run from your desktop/backup folder with: python3 test_terminal.py
## tests every function in mathBack.py (terminal version)
## AI calls (AI_LT, AI_Poly) make REAL API calls — needs .env with OPEN_AI_KEY
## each test prints PASS or FAIL with the result

import sys
import os
import numpy as np
from unittest.mock import patch
from io import StringIO

import mathBack as term

passed = 0
failed = 0

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def capture(func, inputs):
    """Run func() with faked input() calls, return stdout as a string."""
    with patch("builtins.input", side_effect=iter(inputs)), \
         patch("sys.stdout", new_callable=StringIO) as mock_out:
        try:
            func()
        except StopIteration:
            pass  # ran out of inputs — fine for early-exit functions
        return mock_out.getvalue()

def capture_with_ai(func, inputs):
    """Run func() with faked input() calls, using the real AI_LT. Returns stdout."""
    with patch("builtins.input", side_effect=iter(inputs)), \
         patch("sys.stdout", new_callable=StringIO) as mock_out:
        try:
            func()
        except StopIteration:
            pass
        return mock_out.getvalue()

def check(name, out, must_contain=None, must_not_contain=None):
    global passed, failed
    out_lower = out.lower()
    if must_contain and must_contain.lower() not in out_lower:
        print(f"  FAIL  {name}")
        print(f"        expected '{must_contain}' in output")
        print(f"        got: {out.strip()[:120]}")
        failed += 1
        return
    if must_not_contain and must_not_contain.lower() in out_lower:
        print(f"  FAIL  {name}")
        print(f"        did NOT expect '{must_not_contain}' in output")
        print(f"        got: {out.strip()[:120]}")
        failed += 1
        return
    snippet = out.strip().replace("\n", " ")[:80]
    print(f"  PASS  {name}  →  {snippet}")
    passed += 1

def check_raises(name, func, inputs, exc_type):
    global passed, failed
    with patch("builtins.input", side_effect=iter(inputs)):
        try:
            func()
            print(f"  FAIL  {name}  (no exception raised)")
            failed += 1
        except exc_type:
            print(f"  PASS  {name}  →  raised {exc_type.__name__} as expected")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}  →  wrong exception: {type(e).__name__}: {e}")
            failed += 1

def check_reader(name, inputs, expected_shape, expected_values=None):
    global passed, failed
    with patch("builtins.input", side_effect=iter(inputs)):
        try:
            A = term.Matric_Reader()
        except Exception as e:
            print(f"  FAIL  {name}  →  crashed: {e}")
            failed += 1
            return
    if A.shape != expected_shape:
        print(f"  FAIL  {name}  →  shape {A.shape}, expected {expected_shape}")
        failed += 1
        return
    if expected_values is not None and not np.allclose(A, expected_values):
        print(f"  FAIL  {name}  →  values wrong:\n{A}")
        failed += 1
        return
    print(f"  PASS  {name}  →  shape={A.shape}, dtype={A.dtype}")
    passed += 1


print("\n" + "="*60)
print("  mathBack.py  terminal backend test suite")
print("="*60)


# ═════════════════════════════════════════════════════════════
# Matric_Reader
# ═════════════════════════════════════════════════════════════
print("\n── Matric_Reader ──")

check_reader("reads 2x2 correctly",
             ["2","2","1 2","3 4"], (2,2), [[1,2],[3,4]])

check_reader("reads 1x1 correctly",
             ["1","1","7"], (1,1), [[7]])

check_reader("fills short row with zeros",
             ["2","3","1 2","3 4"], (2,3))  # row 0 gets zero-padded to 3 cols

check_reader("returns float dtype",
             ["1","1","5"], (1,1))


# ═════════════════════════════════════════════════════════════
# Matrix_Calc
# ═════════════════════════════════════════════════════════════
print("\n── Matrix_Calc ──")

out = capture(term.Matrix_Calc,
              ["2","2","1 2","3 4",
               "2","2","5 6","7 8"])
check("2x2 multiply prints something", out, must_contain="")

out = capture(term.Matrix_Calc,
              ["1","1","6",
               "1","1","7"])
check("1x1 multiply → 42", out, must_contain="42")

out = capture(term.Matrix_Calc,
              ["2","2","1 0","0 1",
               "2","2","3 7","2 5"])
check("identity × B = B prints something", out, must_contain="")

out = capture(term.Matrix_Calc,
              ["2","2","1 0","0 1",
               "1","3","1 2 3"])
check("size mismatch → error message", out, must_contain="cannot be multiplied")


# ═════════════════════════════════════════════════════════════
# Matrix_Act
# ═════════════════════════════════════════════════════════════
print("\n── Matrix_Act ──")

out = capture(term.Matrix_Act, ["2","2","2 1","1 3"])
check("prints determinant", out, must_contain="determinant")
check("prints inverse", out, must_contain="inverse")

out = capture(term.Matrix_Act, ["2","2","1 2","2 4"])
check("singular matrix mentions inverse", out, must_contain="inverse")
check("singular matrix does NOT print a matrix after inverse line",
      out, must_contain="determinant=0")

out = capture(term.Matrix_Act, ["2","2","1 0","0 1"])
check("identity det=1 mentions determinant", out, must_contain="determinant")

out = capture(term.Matrix_Act, ["2","2","0 0","0 0"])
check("zero matrix det=0", out, must_contain="determinant")


# ═════════════════════════════════════════════════════════════
# Matrix_Diag
# ═════════════════════════════════════════════════════════════
print("\n── Matrix_Diag ──")

out = capture(term.Matrix_Diag, ["2","3"])
check("non-square → error", out, must_contain="not square")

out = capture(term.Matrix_Diag, ["2","2","2 1","1 2"])
check("symmetric 2x2 → diagonalizable", out, must_contain="diagnoliz")

out = capture(term.Matrix_Diag, ["2","2","1 1","0 1"])
check("Jordan block → not diagonalizable", out, must_contain="not")

out = capture(term.Matrix_Diag, ["2","2","3 0","0 5"])
check("diagonal matrix → diagonalizable", out, must_contain="diagnoliz")

out = capture(term.Matrix_Diag, ["3","3","1 0 0","0 2 0","0 0 3"])
check("3x3 diagonal → diagonalizable", out, must_contain="diagnoliz")


# ═════════════════════════════════════════════════════════════
# J_Matrix
# ═════════════════════════════════════════════════════════════
print("\n── J_Matrix ──")

out = capture(term.J_Matrix, ["2","2","2 0","0 3"])
check("diagonal 2x2 → prints P", out, must_contain="P")
check("diagonal 2x2 → prints poly", out, must_contain="poly")

out = capture(term.J_Matrix, ["2","2","1 1","0 1"])
check("Jordan block → prints min poly", out, must_contain="min poly")
check("Jordan block → prints Jordan matrix", out, must_contain="Jordan")

out = capture(term.J_Matrix, ["3","3","1 0 0","0 1 0","0 0 2"])
check("3x3 → prints char poly", out, must_contain="poly")


# ═════════════════════════════════════════════════════════════
# NumberVectorInnerProduct
# ═════════════════════════════════════════════════════════════
print("\n── NumberVectorInnerProduct ──")

out = capture(term.NumberVectorInnerProduct, ["1 2 3","4 5 6"])
check("[1,2,3]·[4,5,6] = 32", out, must_contain="32")

out = capture(term.NumberVectorInnerProduct, ["1 0","0 1"])
check("orthogonal → 0.0", out, must_contain="0.0")

out = capture(term.NumberVectorInnerProduct, ["1 0","-1 0"])
check("[1,0]·[-1,0] = -1.0", out, must_contain="-1.0")

check_raises("size mismatch → ValueError",
             term.NumberVectorInnerProduct, ["1 2","1 2 3"], ValueError)


# ═════════════════════════════════════════════════════════════
# NumberVectorIfBase
# ═════════════════════════════════════════════════════════════
print("\n── NumberVectorIfBase ──")

out = capture(term.NumberVectorIfBase, ["2","1 0","0 1"])
check("standard R2 basis → is a base", out, must_contain="base")

out = capture(term.NumberVectorIfBase, ["2","1 2","2 4"])
check("linearly dependent → not a base", out, must_contain="not")

out = capture(term.NumberVectorIfBase, ["3","1 0 0","0 1 0","0 0 1"])
check("standard R3 basis → is a base", out, must_contain="base")


# ═════════════════════════════════════════════════════════════
# Gram_Schmidt
# ═════════════════════════════════════════════════════════════
print("\n── Gram_Schmidt ──")

out = capture(term.Gram_Schmidt, ["2","1 0","1 1"])
check("non-orthogonal 2D → prints result", out, must_contain="")
check("non-orthogonal 2D → no error", out, must_not_contain="dependent")

out = capture(term.Gram_Schmidt, ["2","1 0","2 0"])
check("linearly dependent → error", out, must_contain="dependent")

out = capture(term.Gram_Schmidt, ["3","1 0 0","0 1 0","0 0 1"])
check("standard R3 basis → prints result", out, must_not_contain="dependent")

out = capture(term.Gram_Schmidt, ["2","1 0","1 1 0"])
check("size mismatch → not the same size", out, must_contain="size")


# ═════════════════════════════════════════════════════════════
# MatricVectorInnerProduct
# ═════════════════════════════════════════════════════════════
print("\n── MatricVectorInnerProduct ──")

out = capture(term.MatricVectorInnerProduct,
              ["2","2","1 0","0 1",
               "2","2","1 0","0 1"])
check("Tr(I^t * I) = 2.0", out, must_contain="2.0")

out = capture(term.MatricVectorInnerProduct,
              ["2","2","0 0","0 0",
               "2","2","1 2","3 4"])
check("zero matrix inner product = 0.0", out, must_contain="0.0")


# ═════════════════════════════════════════════════════════════
# MatricVectorIfBase
# ═════════════════════════════════════════════════════════════
print("\n── MatricVectorIfBase ──")

out = capture(term.MatricVectorIfBase,
              ["4",
               "2","2","1 0","0 0",
               "2","2","0 1","0 0",
               "2","2","0 0","1 0",
               "2","2","0 0","0 1"])
check("standard M2x2 basis → is a base", out, must_contain="base")

out = capture(term.MatricVectorIfBase,
              ["2",
               "2","2","1 0","0 0",
               "2","2","0 1","0 0"])
check("only 2 matrices for M2x2 (dim=4) → too few", out, must_contain="smaller")


# ═════════════════════════════════════════════════════════════
# LT functions  (real AI_LT calls)
# ═════════════════════════════════════════════════════════════
print("\n── Linear Transformation functions ──")

# IsLT
out = capture_with_ai(term.IsLT, ["T(x,y) = (2x+y, x-y)"])
check("IsLT valid → is a linear transformation", out, must_contain="linear transformation")

out = capture_with_ai(term.IsLT, ["T(x,y) = (x^2, y)"])
check("IsLT not linear → not a linear transformation", out, must_contain="not")

# IsIso
out = capture_with_ai(term.IsIso, ["T(x,y) = (x+y, x-y)"])
check("IsIso invertible → iso", out, must_contain="iso")

out = capture_with_ai(term.IsIso, ["T(x,y) = (x+y, x+y)"])
check("IsIso singular → not iso", out, must_contain="det")

# RepresentingMatrix
out = capture_with_ai(term.RepresentingMatrix, ["T(x,y) = (2x, y)"])
check("RepresentingMatrix prints matrix", out, must_contain="")
check("RepresentingMatrix no error", out, must_not_contain="not valid")

out = capture_with_ai(term.RepresentingMatrix, ["T(x,y) = (x^2, y)"])
check("RepresentingMatrix not valid → error", out, must_contain="not valid")

# ImTAndKerT
out = capture_with_ai(term.ImTAndKerT, ["T(x,y) = (x, y)"])
check("ImTAndKerT identity → prints Ker", out, must_contain="ker")
check("ImTAndKerT identity → prints Im", out, must_contain="im")

out = capture_with_ai(term.ImTAndKerT, ["T(x,y) = (0, 0)"])
check("ImTAndKerT zero transform → prints Ker", out, must_contain="ker")


# ═════════════════════════════════════════════════════════════
# Complex_Parser
# ═════════════════════════════════════════════════════════════
print("\n── Complex_Parser ──")

def check_parser(name, v, expected):
    global passed, failed
    try:
        result = term.Complex_Parser(v)
        if isinstance(expected, complex):
            ok = abs(result - expected) < 1e-9
        else:
            ok = abs(result - expected) < 1e-9 and not isinstance(result, complex)
        if ok:
            print(f"  PASS  {name}  →  {result}")
            passed += 1
        else:
            print(f"  FAIL  {name}  →  expected {expected}, got {result}")
            failed += 1
    except Exception as e:
        print(f"  FAIL  {name}  →  crashed: {e}")
        failed += 1

check_parser("plain integer",        "3",      3.0)
check_parser("plain float",          "1.5",    1.5)
check_parser("fraction",             "1/2",    0.5)
check_parser("sqrt",                 "sqrt(2)", 2**0.5)
check_parser("bare i",               "i",      1j)
check_parser("negative i",           "-i",     -1j)
check_parser("a+bi form",            "2+3i",   2+3j)
check_parser("a-bi form",            "2-3i",   2-3j)
check_parser("implicit b=1 (1+i)",   "1+i",    1+1j)
check_parser("implicit b=1 (1-i)",   "1-i",    1-1j)
check_parser("implicit b=1 (2+i)",   "2+i",    2+1j)
check_parser("implicit b=1 (4-i)",   "4-i",    4-1j)
check_parser("pure imaginary 2i",    "2i",     2j)
check_parser("sqrt complex",         "sqrt(2)+i", 2**0.5+1j)
check_parser("invalid → defaults 0", "abc",    0.0)


# ═════════════════════════════════════════════════════════════
# Complex inputs — Matric_Reader
# ═════════════════════════════════════════════════════════════
print("\n── Matric_Reader with complex inputs ──")

check_reader("reads complex 2x2",
             ["2","2","1+2i 3","i 4"], (2,2))

check_reader("reads pure imaginary entries",
             ["2","2","i 0","0 i"], (2,2))

check_reader("reads mixed real and complex",
             ["2","2","1 2+i","3 4-i"], (2,2))


# ═════════════════════════════════════════════════════════════
# Complex inputs — Matrix_Calc
# ═════════════════════════════════════════════════════════════
print("\n── Matrix_Calc with complex inputs ──")

out = capture(term.Matrix_Calc,
              ["2","2","1+i 0","0 1",
               "2","2","1-i 0","0 1"])
check("(1+i)(1-i) diagonal → prints result", out, must_contain="")
check("(1+i)(1-i) → no error", out, must_not_contain="cannot be multiplied")

out = capture(term.Matrix_Calc,
              ["1","1","2i",
               "1","1","3i"])
check("2i × 3i = -6 → contains 6", out, must_contain="6")


# ═════════════════════════════════════════════════════════════
# Complex inputs — NumberVectorInnerProduct
# ═════════════════════════════════════════════════════════════
print("\n── NumberVectorInnerProduct with complex inputs ──")

out = capture(term.NumberVectorInnerProduct, ["1+i 0","1-i 0"])
check("(1+i)·(1-i) prints result", out, must_contain="")
check("no crash on complex vectors", out, must_not_contain="invalid")

out = capture(term.NumberVectorInnerProduct, ["i 0","i 0"])
check("i·i = -1 → contains -1", out, must_contain="-1")

out = capture(term.NumberVectorInnerProduct, ["1+2i 3","4 5-i"])
check("mixed complex inner product prints result", out, must_contain="")


# ═════════════════════════════════════════════════════════════
# Complex inputs — NumberVectorIfBase
# ═════════════════════════════════════════════════════════════
print("\n── NumberVectorIfBase with complex inputs ──")

out = capture(term.NumberVectorIfBase, ["2","1+i 0","0 1-i"])
check("complex vectors — prints base verdict", out, must_contain="base")

out = capture(term.NumberVectorIfBase, ["2","i 0","2i 0"])
check("complex linearly dependent → not a base", out, must_contain="not")


# ═════════════════════════════════════════════════════════════
# Complex inputs — Gram_Schmidt
# ═════════════════════════════════════════════════════════════
print("\n── Gram_Schmidt with complex inputs ──")

out = capture(term.Gram_Schmidt, ["2","1+i 0","0 1"])
check("complex GS — prints result", out, must_contain="")
check("complex GS — no dependent error", out, must_not_contain="dependent")

out = capture(term.Gram_Schmidt, ["2","i 0","2i 0"])
check("complex linearly dependent GS → error", out, must_contain="dependent")


# ═════════════════════════════════════════════════════════════
# Complex inputs — Matrix_Diag
# ═════════════════════════════════════════════════════════════
print("\n── Matrix_Diag with complex inputs ──")

# [[0,1],[-1,0]] has eigenvalues i and -i
out = capture(term.Matrix_Diag, ["2","2","0 1","-1 0"])
check("matrix with complex eigenvalues → prints result", out, must_contain="")
check("complex diag — no crash", out, must_not_contain="error")

# ═════════════════════════════════════════════════════════════
# Complex inputs — MatricVectorInnerProduct
# ═════════════════════════════════════════════════════════════
print("\n── MatricVectorInnerProduct with complex inputs ──")

out = capture(term.MatricVectorInnerProduct,
              ["2","2","i 0","0 i",
               "2","2","i 0","0 i"])
check("complex matrix inner product — prints result", out, must_contain="")
check("complex matrix inner product — no crash", out, must_not_contain="invalid")


# ═════════════════════════════════════════════════════════════
# RESULTS
# ═════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  Results: {passed} passed, {failed} failed")
print("="*60 + "\n")
