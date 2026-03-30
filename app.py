import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "mathBack"))  ## mathBackend.py lives inside the mathBack folder

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from mathBackend import (
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

app = FastAPI()

# This allows the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#######################################################################################################################################
# Request models — these define what data each endpoint expects to receive

class TwoMatrices(BaseModel):
    A: list
    B: list

class OneMatrix(BaseModel):
    A: list

class ManyMatrices(BaseModel):
    matrices: list

class TwoVectors(BaseModel):
    V1: list
    V2: list

class ManyVectors(BaseModel):
    vectors: list

class TwoPolys(BaseModel):
    poly1: str
    poly2: str

class ManyPolys(BaseModel):
    polys: list

class Transformation(BaseModel):
    transformation: str


#######################################################################################################################################
# Matrix endpoints

@app.post("/matrix/multiply")
def matrix_multiply(data: TwoMatrices):
    return Matrix_Calc(data.A, data.B)

@app.post("/matrix/info")
def matrix_info(data: OneMatrix):
    return Matrix_Act(data.A)

@app.post("/matrix/diagonalize")
def matrix_diagonalize(data: OneMatrix):
    return Matrix_Diag(data.A)

@app.post("/matrix/jordan")
def matrix_jordan(data: OneMatrix):
    return J_Matrix(data.A)


#######################################################################################################################################
# Vector endpoints

@app.post("/vectors/number/inner")
def number_vector_inner(data: TwoVectors):
    return NumberVectorInnerProduct(data.V1, data.V2)

@app.post("/vectors/number/isbase")
def number_vector_isbase(data: ManyVectors):
    return NumberVectorIfBase(data.vectors)

@app.post("/vectors/gram-schmidt")
def gram_schmidt(data: ManyVectors):
    return Gram_Schmidt(data.vectors)

@app.post("/vectors/matrix/inner")
def matrix_vector_inner(data: TwoMatrices):
    return MatricVectorInnerProduct(data.A, data.B)

@app.post("/vectors/matrix/isbase")
def matrix_vector_isbase(data: ManyMatrices):
    return MatricVectorIfBase(data.matrices)

@app.post("/vectors/poly/inner")
def poly_inner(data: TwoPolys):
    return PolyInnerProduct(data.poly1, data.poly2)

@app.post("/vectors/poly/isbase")
def poly_isbase(data: ManyPolys):
    return PolyIfBase(data.polys)


#######################################################################################################################################
# Linear Transformation endpoints

@app.post("/lt/matrix")
def lt_matrix(data: Transformation):
    return RepresentingMatrix(data.transformation)

@app.post("/lt/kerimg")
def lt_kerimg(data: Transformation):
    A = Build_Representing_Matrix(data.transformation)
    return ImTAndKerT(A)

@app.post("/lt/islt")
def lt_islt(data: Transformation):
    A = Build_Representing_Matrix(data.transformation)
    return IsLT(A)

@app.post("/lt/isiso")
def lt_isiso(data: Transformation):
    A = Build_Representing_Matrix(data.transformation)
    return IsIso(A)


#######################################################################################################################################
# Run with: uvicorn app:app --reload
