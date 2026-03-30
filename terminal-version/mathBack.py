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

def FractionMatrix(A :np.ndarray) -> np.ndarray:
    A_sp=sp.Matrix(A).applyfunc(sp.nsimplify)
    A=np.array(A_sp)
    return A


## parser that takes an input v(as in vector) and checks if its complex number or not, 
##if so it returns as a complex number like a+bi, otherwise just returns pure real=a
def Complex_Parser(v): 
    try:
        v=v.strip()
        if v=="i": ##given lone i, we need to make it 1j for python syntax
            v="1j"
        elif v=="-i": ##same for -i
            v="-1j"
        elif v.endswith("+i") or v.endswith("-i"): ##given a+-i, we need to make it a+-1j for python syntax
            v=v.replace("i","I")
        else:
            v=v.replace("i","*I")
            
        nv=complex(sp.sympify(v))#try to make it a complex number, if its not a complex number, it will raise an error and we will catch it in the except block
        if nv.imag == 0: #if the imaginary part is 0, then its a pure real number, thus we return just the real part, otherwise we return the complex number as is
            return nv.real
        else:
            return nv
    except:
        print(f"Invalid input: '{v}', defaulting to 0")
        return 0.0


#######################################################################################################################################:)
#1)Metrics:
#builder function
#calc multi
#det,REF,eigenvalue and diags
#jordan mtx min poly char poly

## the next 4 functions are purely for matrix actions


## this is an inner function to be used for the sevreal situations where i need to get as an input a matric
def Matric_Reader() -> np.ndarray:
    MatrixA=[]
    A_Matrix_Rows= int(input("Number of rows in the Matrix = ")) ##count rows and coluns so well know the size of mtx
    A_Matrix_Columns= int(input("Number of columns in the Matrix = "))
    print("enter the Matrix values row by row")
    for i in range(A_Matrix_Rows): ##build mtx one row after another
        raw = input(f"Enter row {i+1}: ") ##take the row as a string
        row =list(map(Complex_Parser, raw.split()))#using complex parser to make sure we can take complex numbers input
        ##treat rows as number vectors,String->sperate->nums->list
        if len(row)<A_Matrix_Columns: 
            row.extend([0.0] * (A_Matrix_Columns - len(row))) ##fill with zero if vector length is short 
        MatrixA.append(row) ##add vector into the array
    A=np.array(MatrixA) ##make it a math object 
    return A


#this part is for two matrix multiplication and can also be used for matrix and vector multiplivation
## it lets the user input 2 metrixes
### and checks to see if they can even be multi(columns left=rows right), if so, returns the product C=AB
####(when A=first matrix, B second matrix)
def Matrix_Calc() -> None:
    A=Matric_Reader()
    B=Matric_Reader()
    if A is None or B is None:
        return
    if A.shape[1] != B.shape[0]:## shape[0]=rows, shape[1]= columns, essentiely checking if columnsA=rowsB as needed in mtx nulti
        print("Cannot be multiplied, wrong matrix sizes")
        return
    C=A@B
    FracC=FractionMatrix(C)
    print(FracC)
    

##this part is for basic matrix info such as:
#-REF
#-inverse
#-determinete
def Matrix_Act() -> None:
    A=Matric_Reader()
    A1=sp.Matrix(A) #diffrent imports sp and np cannot share one matrix need 2 matrix
    A2=np.array(A)
    print("The matrix REF is:")
    rref,_=A1.applyfunc(sp.nsimplify).rref() #REF is the reduced row echelon form, and its a sympy function, but it returns a sympy matrix, thus we need to convert it into a numpy array, but before that we need to apply func sp.nsimplify to make sure we get fractions instead of decimals, thus making it more readable for the user
    print(rref)
    ###################### For determinate 
    
    determinant_np=np.linalg.det(A2)
    print("The matrix determinant is:")
    print(determinant_np)
    ###################### For Inverse 
    
    if determinant_np!=0:#if the determinant is not 0, then the matrix is invertible, thus we can calculate the inverse, otherwise we cannot calculate the inverse as it does not exist
        Matrix_Inv=np.linalg.inv(A2)
        print("The matrix inverse is:")
        FracInv=FractionMatrix(Matrix_Inv)
        print(FracInv)
    else:
        print("Matrix doesnt have an inverse,Determinant=0")
    
#this part is for eigenvalues and to see if the matrix can be diagnloize 
##it lets user input, then it checks if matrix can even be diag, using symm,normal, and Rgeo=Ralg
## were not usinf the helper function as we need to know size of rows to check rank and Rgeo.
def Matrix_Diag() -> None:
    MatrixA=[]
    A_Matrix_Rows= int(input("Number of rows in the first Matrix = "))
    A_Matrix_Columns= int(input("Number of columns in the first Matrix = "))
    if A_Matrix_Columns!=A_Matrix_Rows:
        print("Matrix is not square, cannot be diagnlolize")##non square matrix cannot be diag
        return
    print("enter first Matrix values row by row")
    for i in range(A_Matrix_Rows):
        raw = input(f"Enter row {i+1}: ")
        row = np.array(list(map(Complex_Parser, raw.split())))
        if len(row)<A_Matrix_Columns:
            row.extend([0] * (A_Matrix_Columns - len(row)))
        MatrixA.append(row)
    A=np.array(MatrixA)
    
    Is_Symmetrical = np.allclose(A,A.T) ## checks if A=A^t
    Is_Normal = np.allclose(A @ A.T.conj(), A.T.conj() @ A) ##checks if AA̅=A̅A

    if Is_Symmetrical or Is_Normal:
        eigenvalues,eigenvectors=np.linalg.eigh(A) ##in that case the matrix P is either unitary or hermitian matrix
    else:
        eigenvalues,eigenvectors=np.linalg.eig(A) ## P is a reguler plain ol boring inverse yuck
    
    D=np.diag(eigenvalues)
    P=eigenvectors
    
    rank_P = np.linalg.matrix_rank(P) ##using rank of P,if the rank=dim, thus P is inversable, thus can be diag.
    if rank_P == A_Matrix_Rows:
        Is_diag = True
    else:
        Is_diag = False
    
    ## using basic linear logic, check to see if the matrix can even be diagnoized 
    if Is_Symmetrical:
        msg="Matrix is diagnolizabl, and because the matrix is symmetrical, its eigenvectors are orthonormal"
    elif Is_Normal:
          msg="Matrix is diagnolizable, and because the matrix is normal, its eigenvectors are orthonormal"
    elif Is_diag:
        msg="The matrix is diagnlizalble"
    else:
        msg="The matrix is not diagnlizable"
    
    print(msg)
    
    if Is_diag: ## the idea is, a symm or norm matrix is diag, so either way is diag will return true in the case for symm and norm
        print("The diagnol matrix,D, that your matrix is similiar too is:")
        print(D if np.iscomplexobj(D) else FractionMatrix(D)) ## if the matrix is real, print it as a fraction, if its complex just print it as is, because fractions for complex numbers are just a mess
        print("the eigenvectors, and the columns of the matrix,P, that suffice P^-1AP=D, is")
        print(P if np.iscomplexobj(P) else FractionMatrix(P))

#this part will return a given matrix, A, its Jordan form,Characteristic and Minimal Poly
def J_Matrix() -> None:
    A_np=Matric_Reader()
    A_sp = sp.Matrix(A_np.tolist())#sympy and numpy have different matrix objects, thus we need to convert the np array into a list, and then make it a sympy matrix
    P,J=A_sp.jordan_form()
    FracP=P.applyfunc(sp.nsimplify)
    FracJ=J.applyfunc(sp.nsimplify)
    print("the matrix P that suffice P^-1AP=J, is:")
    print(FracP)
    print("the Jordan matrix is:")
    print(J)
    print("the Char poly for A is:") 
    CharPoly=J.charpoly(x)
    print(CharPoly)

##fix problem with complex eigenvalues in char poly


    ## min poly part
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
    
    print("the min poly is")
    print(min_poly)

##logic of the min poly part:
#i can call the jordan block function, it returns a list of all the blocks(the list is filled with matrix), i create a list ,
#  it holds each eigenvalue, and its max block size
#then i go over the block in the list a11=B[0,0]=eigenvalue for block
#and size=B.rows[0]
#thus making a list that saves all the eigenvalues and its biggest block size, 
#if i have 2 same eigenvalues i check which size is bigger, the bigger one i then put in size

#after that i create a string that goes over the array and prints the min poly like 
#(x-eigenvalue)^size


#######################################################################################################################################:)
#2-vector calculations
#-inner product:
## by recieving 2 vectors calculate the multi of them(seperate into polys,mtx,and number vector)
#-if base:
## by recieving 'n' amount of the vectors "suppose dim=n" check if they not linear dependent => base
#can it span???? idk what does it mean past itay

def Vectors() -> None:
    print("Please input what type of vector you have:\n1-Number Vectors\n2-Matrices\n3-Polynomials-Still in works")
    TypeVector=input()
    if TypeVector=="1":
        NumberVectors()
    elif TypeVector=="2":
        MatrixVectors()
    elif TypeVector=="3":
        PolyVectors()
    else:
        print("not an eligible selection")

#menu 
def NumberVectors() -> None:
    print("\n1-Inner Product, \n2-If Base")
    Action=input()
    if Action=="1":
        NumberVectorInnerProduct()
    elif Action=="2":
        NumberVectorIfBase()
    else:
        print("not an eligible selection")


def NumberVectorInnerProduct() -> float | str:
    print("Input the first vector please")
    raw1 = input("Enter vector values: ") ##take the vector as a string
    V1=np.array(list(map(Complex_Parser, raw1.split())))#take as a string->sepreate numbers->make it numbers->turn to list
    print("Input the second vector please")
    raw2 = input("Enter vector values: ") ##take the vector as a string
    V2=np.array(list(map(Complex_Parser, raw2.split())))
    if len(V1)!=len(V2):
        print ("Vectors are not the same size cant do inner product") #an inner product is a function from VxV-->R thus they need to be the same size
        return
    else:
        result=0
        for i in range(len(V1)):
            result+=V1[i]*V2[i].conjugate() ##the inner product for number vectors is the sum of the multi of each value in the vector, but we need to take the conjugate of the second vector as we are working with complex numbers, thus we do v1[i]*v2[i].conjugate()
        print(result)

def NumberVectorIfBase() -> None:
    print("Please input the amount of vectors.Notice the amount of vecotrs is the dimension were supposedly working in")
    Amount=int(input())
    group=[]
    for i in  range(0,Amount):
        raw=input(f"Enter vector {i+1}: ") ##take the vector as a string
        Veci= np.array(list(map(Complex_Parser, raw.split())))#take as a string->sepreate numbers->make it numbers->turn to list
        group.append(Veci)
    VectorMatrix=np.array(group)
    if np.linalg.det(VectorMatrix)==0:
        print("As columns of a metric, the detiminant is, 0, thus the metric singuler, and by Therom 3.10.6, the columns, which are the "
        "vectors do not form a base")
    else:
        print("As columns of a metric, the detiminant!=0, thus the metric ivertible, and by Therom 3.10.6, the columns, which are the "
        "vectors form a base")


def Gram_Schmidt() ->None:
    print("input amount of vectors")
    Amount=int(input())
    group=[]
    for i in range(0,Amount):
        raw=input(f"Enter vector {i+1}: ")
        Veci= np.array(list(map(Complex_Parser, raw.split()))) #like above, the way to get input of vectors
        if i>0 and len(Veci)!=len(group[0]): ##check if the new inoput vector is the same length as the first one
            print("not the same size")
            return
        vi=sp.Matrix(Veci) ##define each vector as a matrix for sympy
        group.append(vi) ##add the matrix to the group
    orth=sp.Matrix.orthogonalize(*group) ##unpack the list
    if not orth: ## if list is emprt, orth didnt work because they are LD
        print("vectors are linear dependetnt cannot orthogonalize")
    elif len(orth)<len(group): ##one of the vectors are linear dependent thus cannot ort all of them
        print("at least one of the vectors is lineary dependent, cannot orthogonalize ")
    else:
        print(orth)
    
##menu for the matric vector stuff
def MatrixVectors() -> None:
    print("\n1-Inner Product, \n2-If Base")
    Action=input()
    if Action=="1":
        MatricVectorInnerProduct()
    elif Action=="2":
       MatricVectorIfBase()
    else:
        print("not an eligible selection")


## get 2 matrics, multi on B^t,A, calculate trace of that mtx
def MatricVectorInnerProduct() -> float: 
    A=Matric_Reader()
    B=Matric_Reader()
    if A is None or B is None:
        print ("Invalid matrix input")

    if A.shape != B.shape:
         print("Matrices must have the same dimensions")
    C=B.T @ A
    result= float(np.trace(C)) ##(A,B)=Tr(B^t*A) def of mtx inner prodct
    print(result)

## recieve n amount of mtx, using the standard base E, we can make every mtx a vector, and then make an mtx, calc det
def MatricVectorIfBase() -> str: ##chat did that function
    print("Please input the amount of mtx, they all have to be the same size ")
    Group=[]
    Amount=int(input())
    for i in range(0,Amount):
        print(f"\nMatrix {i+1:}") ##take as an input all the dumbass metrics the dumb user wants to check
        Ai=Matric_Reader()

        if Ai is None:  ##if mf put NOTHING we blasting his bitch ass
            return "invalid matrix input"
        if i>0 and Ai.shape!=Group[0].shape:  ##if those damn mf mtx are not the same damn size we got nothing to check 
            return "All matrices must have the same dimensions"
        Group.append(Ai) ## add your bitch ass mtx into the group
        
    rows,cols=Group[0].shape ## if they all the same size we gotta check the overall dim of the space and remember DimMnxn=n^2
    Dim=rows*cols
    if Amount<Dim:
        print("Cannot form a base if group cardinlity is smaller than our dim")
        return
    Make_Vector=[Ai.flatten() for Ai in Group] #instead of a mtx make it a number vector (basicaly a loop that goes through the group array
    # and makes each mtx in the group a number)
    Huge_Ass_Mtx=np.array(Make_Vector).T ## build a mtx from all the number vectors we gatherd
    if np.linalg.det(Huge_Ass_Mtx)==0: ##basic linear dummy
        print ("not a base, representing matrix det=0, thus singuler, so cannot be a base")
    else:
        print ("the group of matrices is a base, the representing matrix det!=0. thus they make a base")


def PolyVectors() -> None:
    print("\n1-Inner Product\n2-If Base")
    Action=input()
    if Action=="1":
        PolyInnerProduct()
    elif Action=="2":
        PolyIfBase()
    else:
        print("not an eligible selection") 


def PolyInnerProduct() -> None: 
    print("Enter your first polynomial:") 
    User_Input1=str(input())
    P=AI_Poly(User_Input1) ##using ai to make the poly input comftorable for user and for sympy demands
    print("Enter your second polynomial:")
    User_Input2=str(input())
    Q=AI_Poly(User_Input2) ##same here
    result=sp.integrate(P*Q,(x,0,1)) ##calculate the inner product
    print(result)
    

def Poly_To_Vector(P:sp.Expr, dim:int) -> list: ##chat did
    Poly=sp.Poly(P,x)

    coeffs = Poly.all_coeffs()      # highest degree → constant
    coeffs = list(reversed(coeffs)) # constant → highest degree → as this the standard way for polys in base E

    while len(coeffs) < dim: ##fill zero if emprt ex:dim=3, P=1+x → vecP=(1,1,0)
        coeffs.append(0)

    return coeffs[:dim]

def PolyIfBase() ->None:
    Group=[]
    print("please enter the amoount of polynomials,keep in mind, the amount of polynomials is the dimesnsion were supposedly working in")
    dim=int(input())
    for i in range(dim):
        print(f"\nPolynomial {i+1:}")
        User_Poly=str(input())
        Pi=AI_Poly(User_Poly)
        
        if Pi is None:
            return "invalid Poly input"
        if sp.degree(Pi,x)>=dim:
            return "Poly degree is higher than the dim, not a valid input"
        Group.append(Pi)
    CoaVectors=[Poly_To_Vector(Pi,dim) for Pi in Group] ##build vectors from poly
    Mtx=sp.Matrix(CoaVectors) ##make a mtx from said vectors
    if Mtx.det()!=0: ##classic linear 
        print ("the representing mtx for this group, using base E,detirmaniate is not 0, thus, its a base")
    else:
        print ("not a base")





    
#######################################################################################################################################:)
#3-Linear transformation
#if is a linear transformation 
#find Ker and Im
#find its matrix. and by receivng a base, 1) make sure its a base using function from before, 2)calculate the mtx with that base
##the idea would be using mtx by E, and calculate P^-1[(T]E)P, when P columns are said base
#is isomorphic
#have the option after finding mtx,to use any of the first 4 functions
def LinearTransformationMenu() ->None:
    print("Please select your action: \n1-Representing Matrix \n2-ImT and KerT \n3-Is A Linear Trasnformation  \n4-Is Isomorphic" \
    "\nPlease notice, as for now, this function only works in F^n linear space(only number vector)")
    Action=input()
    if Action=="1":
        RepresentingMatrix()
    elif Action=="2":
       ImTAndKerT()
    elif Action=="3":
        IsLT()
    elif Action=="4":
        IsIso()
    else:
        print("not an eligible selection, stop fucking around shit for brains")


def RepresentingMatrix() -> None:
    T=Build_Representing_Matrix()
    if T is None:
        print("the Transformation is not valid, thus no matrix")
    else:
        FracT=FractionMatrix(T)
        print(FracT)

def Build_Representing_Matrix() -> np.ndarray | None: 
    print("please enter the linear transformation")
    UserTransformation=str(input())
    if UserTransformation.lower()=="t=x,y,z":
            UserTransformation="T(x,y,z)=(x,y,z)"
    elif UserTransformation.lower()=="t=x,y":
            UserTransformation="T(x,y)=(x,y)"
    T=AI_LT(UserTransformation)
    return T       ##AI_LT job is to return the rep mtx by E
   

def ImTAndKerT() -> None:  
    A=Build_Representing_Matrix()
    A_sp=sp.Matrix(A)
    Ker=A_sp.nullspace()
    Img=A_sp.columnspace()
    print("Ker(T) =")

    if not Ker:
        print("{0}") ##if ker is empty print 0
    else:
        for v in Ker:
            sp.pprint(v)## if not empty print in a nice way av columns vectors
            print()

    print("\nIm(T) =")

    for v in Img: ##same for img
        sp.pprint(v)
        print()


def IsLT() ->None:
    A=Build_Representing_Matrix() ##in AI_LT, if it returns a mtx it also means the input is LT
    if A is None: ## if its none it means chat couldnt build a mtx, thus concluding its not a LT
        print("not a linear transformation")
    else:
        print("is a linear transformation")

def IsIso() -> None:
    A=Build_Representing_Matrix()
    if A is None:
        raise ValueError("the given LT is not a valid LT")
    if A.shape[0]==A.shape[1]:##metric is square thus could be iso
        if np.linalg.det(A)!=0:
            print("represebnting metric det!=0,thus reguler,thus T is iso")
        else:
            print("no becuase metric det=0")
    else:
        print("when T:V-->U, the dimensions between V and U differ, thus could not be iso")



#######################################################################################################################################:)
##menu 

def main():
    print("Welcome to linear algebra helper,please choose on of the following:\n1-Matrix multiplication\n2-Matrix info\n3-" \
    "Matrix Diafnlizition\n4-Jordan form and Characteristic Polynomial\n5-Vectors Math\n6-Linear Transformations\n7-GS")
    selction=input()
    if selction=="1":
        Matrix_Calc()
    elif selction=="2":
        Matrix_Act()
    elif selction=="3":
        Matrix_Diag()
    elif selction=="4":
        J_Matrix()
    elif selction=="5":
        Vectors()
    elif selction=="6":
        LinearTransformationMenu()
    elif selction=="7":
        Gram_Schmidt()
    else:
        print("Not an elgible selection")
if __name__ == "__main__":
    main()

