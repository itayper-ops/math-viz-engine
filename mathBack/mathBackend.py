#add libraries 
import numpy as np
import math 
import sympy as sp 
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from sympy import symbols
from fractions import Fraction


#linear algebra:
##explain the project doofus
##this here is a calculator for linear algebra, a lot of stuff here i wished i had when i was cramming for linear 2,
##but it doesnt matter, as of the day that im writing this ive learned that i got 87 on my test
##either way, this is the backend part, its the mind of the project
##i have 2 AI, API calls as, getting an input of LT and Polys are pretty exhusting and doesnt let the user have freedom on how he writes
##the project is spereated into:
#1-background AI functions
#2-Matrix calculations and functions
#3-vector calculations for number vectors,polys and of course matrix
#4-LT stuff.

#######################################################################################################################################:)
## this part is the AI function, its an inside function that gets called from poly and LT function,it recievs input from a user,
# reads and checks if the input is valid to the function that called it
#returns the input in a clean way that would be easy to use in said functions
load_dotenv()
Chat=OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

def AI_LT(user_input: str) -> np.ndarray | None:
    prompt=f"""You are a linear algebra assistant.

    The user gives a transformation.
    Determine whether it is linear.
    If it is linear, compute its representing matrix in the standard basis.

    Return ONLY valid JSON.
    Do not write explanations.
    Do not use markdown.
    Do not use code blocks.
    If linear, return exactly:
    {{"matrix": [[...], [...], ...]}}

    If not linear, return exactly:
    {{"matrix": null}} 
    user question=
    {user_input}"""
    response=Chat.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    Answer=response.output_text.strip() ##returns the ai generated text
    Data=json.loads(Answer)## converts it into a json file
    Matrix=Data.get("matrix") #grab whats under the matrix header

    if Matrix is None:
        return None
    Te=np.array(Matrix,dtype=float)
    return Te


x= sp.symbols('x')

def AI_Poly(user_input:str) -> sp.Expr:
    prompt=f"""you are a math assistant.
    The user gives a polynomial.
    Determine whether the polynomial is a Continuous function on the interval [0,1]
    if it is,Rewrite it in valid SymPy syntax.
    Use only the variable x.
    Return ONLY valid JSON.
    Do not write explanations.
    Do not use markdown.
    Do not use code blocks.

    If the polynomial is a continuous function on the interval [0,1] Return exactly in this format:
    {{"Polynomial":"..."}} 

    if the polynomial is not a continuous function on the interval [0,1] Return excatly:
    {{"Polynomial":"null"}}
    User question=
    {user_input}"""
    response=Chat.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    Answer=response.output_text.strip()
    Data=json.loads(Answer)
    P=Data.get("Polynomial")
    return sp.sympify(P, locals={"x":x})

def FractionMatrix(A: np.ndarray) -> list:
    A_sp=sp.Matrix(A).applyfunc(sp.nsimplify)
    return [[str(v) for v in row] for row in A_sp.tolist()]

## parser that takes an input v(as in vector) and checks if its complex number or not, 
##if so it returns as a complex number like a+bi, otherwise just returns pure real=a
def Complex_Parser(v): 
    try:
        v=v.strip()
        if v=="i":
            v="1j"
        elif v=="-i":
            v="-1j"
        elif v.endswith("+i") or v.endswith("-i"):
            v=v.replace("i","I")
        else:
            v=v.replace("i","*I")
            
        nv=complex(sp.sympify(v))
        if nv.imag == 0:
            return nv.real
        else:
            return nv
    except:
        return 0.0

##to fix a+-i problem doing v.replace("i",I) is enough 

def Complex_Mat(A: list) -> list:
    return [[Complex_Parser(str(v)) for v in row] for row in A]

#######################################################################################################################################:)
#1)Metrics:
#calc multi
#det,REF,eigenvalue and diags
#jordan mtx min poly char poly

## the next 4 functions are purely for matrix actions

#this part is for two matrix multiplication and can also be used for matrix and vector multiplivation
## it takes 2 metrixes as lists
### and checks to see if they can even be multi(columns left=rows right), if so, returns the product C=AB
####(when A=first matrix, B second matrix)
def Matrix_Calc(A: list, B: list) -> dict:
    A=np.array(Complex_Mat(A))
    B=np.array(Complex_Mat(B))
    if A.shape[1] != B.shape[0]:## shape[0]=rows, shape[1]= columns, essentiely checking if columnsA=rowsB as needed in mtx nulti
        return {"error": "Cannot be multiplied, wrong matrix sizes"}
    C=A@B
    FracC=FractionMatrix(C)
    return {"result": FracC}


##this part is for basic matrix info such as:
#-REF
#-inverse
#-determinete
def Matrix_Act(A: list) -> dict:
    A1=sp.Matrix(A) #diffrent imports sp and np cannot share one matrix need 2 matrix
    A2=np.array(Complex_Mat(A))

    rref,_=A1.applyfunc(sp.nsimplify).rref()
    rref_out=[[str(v) for v in row] for row in rref.tolist()]

    ###################### For determinate 
    determinant_np=float(np.linalg.det(A2))

    result={
        "rref": rref_out,
        "determinant": determinant_np
    }

    ###################### For Inverse 
    if abs(determinant_np)>1e-10: ##using abs because floating point can give something like 1e-17 instead of 0
        Matrix_Inv=np.linalg.inv(A2)
        FracInv=FractionMatrix(Matrix_Inv)
        result["inverse"]=FracInv
    else:
        result["inverse"]=None ##Matrix doesnt have an inverse,Determinant=0

    return result

    
#this part is for eigenvalues and to see if the matrix can be diagnloize 
##it takes the matrix as a list, then it checks if matrix can even be diag, using symm,normal, and Rgeo=Ralg
def Matrix_Diag(A: list) -> dict:
    A=np.array(Complex_Mat(A))
    rows,cols=A.shape

    if rows!=cols:
        return {"error": "Matrix is not square, cannot be diagnlolize"}##non square matrix cannot be diag

    Is_Symmetrical = np.allclose(A,A.T) ## checks if A=A^t
    Is_Normal = np.allclose(A @ A.T.conj(), A.T.conj() @ A) ##checks if AA̅=A̅A

    if (Is_Symmetrical or Is_Normal) and not np.iscomplexobj(A): ##if its symm or normal and real, we can use eigh which is more accurate for those cases, otherwise we have to use eig
        eigenvalues,eigenvectors=np.linalg.eigh(A) ##in that case the matrix P is either unitary or hermitian matrix
    else:
        eigenvalues,eigenvectors=np.linalg.eig(A) ## P is a reguler plain ol boring inverse yuck
    
    D=np.diag(eigenvalues)
    P=eigenvectors
    
    rank_P = np.linalg.matrix_rank(P) ##using rank of P,if the rank=dim, thus P is inversable, thus can be diag.
    Is_diag = rank_P == rows
    
    ## using basic linear logic, check to see if the matrix can even be diagnoized 
    if Is_Symmetrical:
        msg="Matrix is diagnolizabl, and because the matrix is symmetrical, its eigenvectors are orthonormal"
    elif Is_Normal:
        msg="Matrix is diagnolizable, and because the matrix is normal, its eigenvectors are orthonormal"
    elif Is_diag:
        msg="The matrix is diagnlizalble"
    else:
        msg="The matrix is not diagnlizable"

    result={"message": msg, "is_diagonalizable": bool(Is_diag)}

    if Is_diag: ## the idea is, a symm or norm matrix is diag, so either way is diag will return true in the case for symm and norm
        ## if the matrix is real, print it as a fraction, if its complex just convert to string as is, because fractions for complex numbers are just a mess
        result["D"]=FractionMatrix(D) if not np.iscomplexobj(D) else [[str(v) for v in row] for row in D.tolist()]
        result["P"]=FractionMatrix(P) if not np.iscomplexobj(P) else [[str(v) for v in row] for row in P.tolist()]

    return result


#this part will return a given matrix, A, its Jordan form,Characteristic and Minimal Poly
def J_Matrix(A: list) -> dict:
    A_np=np.array(Complex_Mat(A))
    A_np=np.round(A_np, 10) ##kill floating point noise before passing to sympy, otherwise jordan_form hangs forever
    A_sp=sp.Matrix(A_np.tolist()) #sympy and numpy have different matrix objects, thus we need to convert the np array into a list, and then make it a sympy matrix
    P,J=A_sp.jordan_form()
    FracP=[[str(v) for v in row] for row in P.applyfunc(sp.nsimplify).tolist()]
    FracJ=[[str(v) for v in row] for row in J.applyfunc(sp.nsimplify).tolist()]

    CharPoly=A_sp.charpoly(x)

    ## min poly part
    ##logic: i can call the jordan block function, it returns a list of all the blocks(the list is filled with matrix), i create a list ,
    ##  it holds each eigenvalue, and its max block size
    ##then i go over the block in the list a11=B[0,0]=eigenvalue for block
    ##and size=B.rows[0]
    ##thus making a list that saves all the eigenvalues and its biggest block size, 
    ##if i have 2 same eigenvalues i check which size is bigger, the bigger one i then put in size
    ##after that i create a string that goes over the array and prints the min poly like 
    ##(x-eigenvalue)^size
    Blocks=J.get_diag_blocks()    ##sympy function to return jordan matrix bloc as a list
    biggest={} ##make a list that will hold each eigenvalue and its biggest block size
    for B in Blocks: ##go over the jordan block list one matrix at a time
        lam=B[0,0] ##the first value in a jordan matrix is his eigenvalue
        size=B.shape[0] ##shape.[0]=rows, and each block is a square matrix thus its size=rows

        if lam not in biggest: ## if the eigenvalue is not in biggest --> thus a new eigenvalue we encounterd 
            biggest[lam]=size ##new eigenvalue save its size next the eigenvalue 
        else:
            biggest[lam] = max(biggest[lam],size) ##we already saved the eigenvalue, check which block size is bigger

       ## building the min poly 
    min_poly = 1  
    for lam, size in biggest.items():  
        min_poly *= (x - lam)**size

    return {
        "P": FracP,
        "J": FracJ,
        "char_poly": str(CharPoly),
        "min_poly": str(min_poly)
    }


#######################################################################################################################################:)
#2-vector calculations
#-inner product:
## by recieving 2 vectors calculate the multi of them(seperate into polys,mtx,and number vector)
#-if base:
## by recieving 'n' amount of the vectors "suppose dim=n" check if they not linear dependent => base

def NumberVectorInnerProduct(V1: list, V2: list) -> dict:
    V1=np.array(list(map(Complex_Parser, [str(v) for v in V1])))
    V2=np.array(list(map(Complex_Parser, [str(v) for v in V2])))
    if len(V1)!=len(V2):
        return {"error": "Vectors are not the same size cant do inner product"} #an inner product is a function from VxV-->R thus they need to be the same size
    result=0
    for i in range(len(V1)):
        result+=V1[i]*complex(V2[i]).conjugate() ##the inner product for number vectors is the sum of the multi of each value in the vector, but we need to take the conjugate of the second vector as we are working with complex numbers, thus we do v1[i]*v2[i].conjugate()
        ## wrapping in complex() because plain floats dont have .conjugate() in numpy
    result=complex(result)
    return {"result": str(result) if result.imag!=0 else float(result.real)}

def NumberVectorIfBase(vectors: list) -> dict:
    group=[]
    for v in vectors:
        parsed=np.array(list(map(Complex_Parser, [str(x) for x in v])))
        group.append(parsed)
    VectorMatrix=np.array(group)
    if VectorMatrix.shape[0]!=VectorMatrix.shape[1]:
        return {"error": "Amount of vectors must equal their dimension to form a base"}
    if abs(complex(np.linalg.det(VectorMatrix)))<1e-10: ##basic linear dummy, using abs because det could be complex
        return {"result": False, "message": "As columns of a metric, the detiminant is 0, thus the metric singuler, and by Therom 3.10.6, the columns, which are the vectors do not form a base"}
    else:
        return {"result": True, "message": "As columns of a metric, the detiminant!=0, thus the metric ivertible, and by Therom 3.10.6, the columns, which are the vectors form a base"}


def Gram_Schmidt(vectors: list) -> dict:
    group=[]
    for i, vec in enumerate(vectors):
        parsed=list(map(Complex_Parser, [str(v) for v in vec]))
        if i>0 and len(parsed)!=len(vectors[0]): ##check if the new inoput vector is the same length as the first one
            return {"error": "not the same size"}
        vi=sp.Matrix(parsed) ##define each vector as a matrix for sympy
        group.append(vi) ##add the matrix to the group
    orth=sp.Matrix.orthogonalize(*group) ##unpack the list
    if not orth: ## if list is emprt, orth didnt work because they are LD
        return {"error": "vectors are linear dependetnt cannot orthogonalize"}
    elif len(orth)<len(group): ##one of the vectors are linear dependent thus cannot ort all of them
        return {"error": "at least one of the vectors is lineary dependent, cannot orthogonalize"}
    else:
        def fmt(v): ##helper to handle complex vs real output
            c=complex(v)
            return str(c) if c.imag!=0 else float(c.real)
        return {"result": [[fmt(v) for v in vec] for vec in orth]}


## get 2 matrics, multi on B^t,A, calculate trace of that mtx
def MatricVectorInnerProduct(A: list, B: list) -> dict:
    A=np.array(Complex_Mat(A))
    B=np.array(Complex_Mat(B))
    if A.shape != B.shape:
        return {"error": "Matrices must have the same dimensions"}
    C=B.T @ A
    result=np.trace(C) ##(A,B)=Tr(B^t*A) def of mtx inner prodct
    return {"result": str(result) if np.iscomplexobj(result) else float(result)}

## recieve n amount of mtx, using the standard base E, we can make every mtx a vector, and then make an mtx, calc det
def MatricVectorIfBase(matrices: list) -> dict:
    Group=[np.array(Complex_Mat(m)) for m in matrices] ##take as an input all the dumbass metrics the dumb user wants to check
    Amount=len(Group)

    for i in range(1, Amount):
        if Group[i].shape!=Group[0].shape:  ##if those damn mf mtx are not the same damn size we got nothing to check 
            return {"error": "All matrices must have the same dimensions"}
        
    rows,cols=Group[0].shape ## if they all the same size we gotta check the overall dim of the space and remember DimMnxn=n^2
    Dim=rows*cols
    if Amount<Dim:
        return {"error": "Cannot form a base if group cardinlity is smaller than our dim"}
    Make_Vector=[Ai.flatten() for Ai in Group] #instead of a mtx make it a number vector (basicaly a loop that goes through the group array
    # and makes each mtx in the group a number)
    Huge_Ass_Mtx=np.array(Make_Vector).T ## build a mtx from all the number vectors we gatherd
    if abs(complex(np.linalg.det(Huge_Ass_Mtx)))<1e-10: ##basic linear dummy
        return {"result": False, "message": "not a base, representing matrix det=0, thus singuler, so cannot be a base"}
    else:
        return {"result": True, "message": "the group of matrices is a base, the representing matrix det!=0. thus they make a base"}


def PolyInnerProduct(poly1: str, poly2: str) -> dict:
    P=AI_Poly(poly1) ##using ai to make the poly input comftorable for user and for sympy demands
    Q=AI_Poly(poly2) ##same here
    result=sp.integrate(P*Q,(x,0,1)) ##calculate the inner product
    return {"result": str(result)}


def Poly_To_Vector(P:sp.Expr, dim:int) -> list:
    Poly=sp.Poly(P,x)
    coeffs = Poly.all_coeffs()      # highest degree → constant
    coeffs = list(reversed(coeffs)) # constant → highest degree → as this the standard way for polys in base E
    while len(coeffs) < dim: ##fill zero if emprt ex:dim=3, P=1+x → vecP=(1,1,0)
        coeffs.append(0)
    return coeffs[:dim]

def PolyIfBase(polys: list) -> dict:
    Group=[]
    dim=len(polys) ##the amount of polynomials is the dimesnsion were supposedly working in
    for poly_str in polys:
        Pi=AI_Poly(poly_str)
        if Pi is None:
            return {"error": "invalid Poly input"}
        if sp.degree(Pi,x)>=dim:
            return {"error": "Poly degree is higher than the dim, not a valid input"}
        Group.append(Pi)
    CoaVectors=[Poly_To_Vector(Pi,dim) for Pi in Group] ##build vectors from poly
    Mtx=sp.Matrix(CoaVectors) ##make a mtx from said vectors
    if Mtx.det()!=0: ##classic linear 
        return {"result": True, "message": "the representing mtx for this group, using base E,detirmaniate is not 0, thus, its a base"}
    else:
        return {"result": False, "message": "not a base"}


#######################################################################################################################################:)
#3-Linear transformation
#if is a linear transformation 
#find Ker and Im
#find its matrix. and by receivng a base, 1) make sure its a base using function from before, 2)calculate the mtx with that base
##the idea would be using mtx by E, and calculate P^-1[(T]E)P, when P columns are said base
#is isomorphic

def Build_Representing_Matrix(UserTransformation: str) -> np.ndarray | None:
    if UserTransformation.lower()=="t=x,y,z":
        UserTransformation="T(x,y,z)=(x,y,z)"
    elif UserTransformation.lower()=="t=x,y":
        UserTransformation="T(x,y)=(x,y)"
    T=AI_LT(UserTransformation)
    return T       ##AI_LT job is to return the rep mtx by E

def RepresentingMatrix(UserTransformation: str) -> dict:
    T=Build_Representing_Matrix(UserTransformation)
    if T is None:
        return {"error": "the Transformation is not valid, thus no matrix"}
    FracT=FractionMatrix(T)
    return {"result": FracT}

def ImTAndKerT(A: np.ndarray) -> dict:
    if A is None:
        return {"error": "invalid transformation"}
    A_sp=sp.Matrix(A)
    Ker=A_sp.nullspace()
    Img=A_sp.columnspace()
    return {
        "Ker": [[str(v) for v in vec] for vec in Ker], ##if ker is empty list is empty, frontend handles the {0} case
        "Img": [[str(v) for v in vec] for vec in Img]
    }

def IsLT(A: np.ndarray) -> dict:
    if A is None: ## if its none it means chat couldnt build a mtx, thus concluding its not a LT
        return {"error": "not a linear transformation"}
    return {"result": "is a linear transformation"}

def IsIso(A: np.ndarray) -> dict:
    if A is None:
        return {"error": "the given LT is not a valid LT"}
    if A.shape[0]==A.shape[1]:##metric is square thus could be iso
        if np.linalg.det(A)!=0:
            return {"result": "represebnting metric det!=0,thus reguler,thus T is iso"}
        else:
            return {"error": "no becuase metric det=0"}
    else:
        return {"error": "when T:V-->U, the dimensions between V and U differ, thus could not be iso"}
