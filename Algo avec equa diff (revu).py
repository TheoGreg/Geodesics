import sympy as sp
import numpy as np
from scipy.integrate import odeint



##Affectations

x, y, z = sp.symbols("x, y, z") #Coordonnees

# f = (x**2 + y**2 + z**2 + 12)**2 - 64 * (x**2 + y**2) # Surface

f = (x**2 + y**2 + z**2) - 4

def grad(f, coord):
    """ Donne le gradient de f(coord) """
    gradient = []
    for a in coord:
        gradient.append(sp.diff(f, a))
    return(sp.Matrix(gradient))
    
G = grad(f, [x, y, z]) # Gradient de f

H = sp.hessian(f, [x, y, z]) # Hessienne de f

def dH(f, coord):
    H = list(sp.hessian(f, coord))
    for i in range(len(coord)):
        for j in range(len(coord)):
            d = []
            for a in coord:
                d.append(sp.diff(H[i*len(coord)+j], a))
            H[i*len(coord)+j] = d
    return np.reshape(np.array(H), (3, 3, 3))
    
I = dH(f, [x, y, z]) 

vx, vy, vz = sp.symbols("vx, vy, vz") # Coordonnees du vecteur vitesse instantanee de la gedodesique
V = sp.Matrix([vx, vy, vz])

px, py, pz = sp.symbols("px, py, pz") # Coordonnees de la derivee par rapport a h
P = sp.Matrix([px, py, pz])

pvx, pvy, pvz = sp.symbols("pvx, pvy, pvz") # Coordonnees du vecteur vitesse instantanee de la derivee par rapport a h
PV = sp.Matrix([pvx, pvy, pvz])

## Equations

equa1 = sp.lambdify((x,y,z,vx,vy,vz), -(G /(G.T*G)[0]) * (V.T * H * V)[0])

special = sp.Matrix( np.dot(I, np.array([px, py, pz])))

equa2 = sp.lambdify((x, y, z, vx, vy, vz, px, py, pz, pvx, pvy, pvz), 2* (G * G.T * H * P)/ ((G.T * G)[0])**2 - (H * P)/(G.T * G)[0]*(V.T * H * V)[0] - (G / (G.T * G)[0]) * ((V.T * special * V) + (V.T * H * PV) + (PV.T * H * V))[0])

## Resolution

def euler(f, Y0, T):
    solution = [np.array(Y0)] * len(T)
    for i in range(0, len(T)-1):
        solution[i+1] = solution[i] + (T[i+1] - T[i]) * np.array(f(solution[i], T[i]))
    return(solution)
   
## Algorithmes

def geodesique(depart, arrivee, N, epsilon):
    
    #Création de la fonction gradient
    
    fonction_grad = sp.lambdify((x,y,z), G)
    gradient = fonction_grad(depart[0],depart[1],depart[2])
    
    #Choix d'une base de vecteurs tangents à la surface au point de départ
    
    if gradient[0] > 0.000001 :
        V1 = [gradient[2]/100, 0, - gradient[0]/100]
        V2 = [gradient[1]/100, -gradient[0]/100, 0]
    elif gradient[1] > 0.000001 :
        V1 = [0, gradient[2]/100, - gradient[1]/100]
        V2 = [gradient[1]/100, -gradient[0]/100, 0]
    else : 
        V1 = [0, gradient[2]/100, - gradient[1]/100]
        V2 = [gradient[2]/100, 0, - gradient[0]/100]
    
    #Création du temps
    
    T = np.linspace(0, 1, N)
    
    #Initialisation du nombre de géodésiques calculées
    
    iter = 1
    
    #Introduction des deux équations différentielles
    
    def f1(X, t):
        """fonction donnant la ligne géodésique grâce à (x'', y'', z'') a partir de (x, y, z, x', y', z') et de t"""
        A = equa1(X[0], X[1], X[2], X[3], X[4], X[5])
        return([X[3], X[4], X[5], A[0], A[1], A[2]])
        
    def f2(X, t):
        """fonction calculant la dérivée de F_intermédiaire selon un vecteur de la base des vitesses tangentes"""
        A = equa2(chemin[int(t * (N-1))][0], chemin[int(t * (N-1))][1], chemin[int(t * (N-1))][2], chemin[int(t * (N-1))][3], chemin[int(t * (N-1))][4], chemin[int(t * (N-1))][5], X[0], X[1], X[2], X[3], X[4], X[5])
        return([X[3], X[4], X[5], A[0], A[1], A[2]])
    
    #Choix d'un vecteur vitesse initiale 
    
    V = [V1[0] + V2[0], V1[1] + V2[1], V1[2] + V2[2]]
    
    #Calcul de la géodésique 
    
    chemin = odeint(f1, depart + V, T)
    
    print("La vitesse initiale est")
    print(V)
    
    #Calcul du point d'arrivée de la géodésique
    
    fin = sp.Matrix([chemin[-1][0], chemin[-1][1], chemin[-1][2]])
    
    print(fin)
    
    #Calcul de l'écart entre le point d'arrivée et le point désiré
    
    ecart = sp.sqrt(((fin - sp.Matrix(arrivee)).T * (fin - sp.Matrix(arrivee)))[0])
    
    print("L'écart vaut")
    print(ecart)
    
    while ecart > epsilon :
        
        #Affichage de la géodésique obtenue après 3 itérations
        
        # if iter%3 == 1:
        #     affichage(triangles, chemin)
        
        print("L'écart vaut")
        print(ecart)
        
        # def f2(X, t):
        #     A = equa_schwarz(chemin[int(t * (N-1))][0], chemin[int(t * (N-1))][1], chemin[int(t * (N-1))][2], chemin[int(t * (N-1))][3], chemin[int(t * (N-1))][4], chemin[int(t * (N-1))][5], X[0], X[1], X[2], X[3], X[4], X[5])
        #     return([X[3], X[4], X[5], A[0], A[1], A[2]])
        
        # print("checkpoint1")
        
    
        #Calcul de la dérivée de F_intermédiaire selon V1 (dernière valeur)
        
        variations1 = euler(f2, [0, 0, 0] + V1, T)
        
        #Calcul de la dérivée de F_intermédiaire selon V2 (dernière valeur)
        
        variations2 = euler(f2, [0, 0, 0] + V2, T)
        
        #Calcul de la dérivée de F selon V1 
        
        derivee1 = ((sp.Matrix(variations1[-1][0:3])).T * (fin - sp.Matrix(arrivee)))[0] / ecart
        
        #Calcul de la dérivée de F selon V2
        
        derivee2 = ((sp.Matrix(variations2[-1][0:3])).T * (fin - sp.Matrix(arrivee)))[0] / ecart
        
        #Calcul du facteur nécessaire à la méthode du gradient (WTF ecart)
        
        facteur = ecart / (derivee1**2 + derivee2**2)
        
        #Modification du vecteur vitesse initiale avec la méthode du gradient
        
        V[0] = V[0] - (derivee1*V1[0] - derivee2*V2[0])*facteur
        V[1] = V[1] - (derivee1*V1[1] - derivee2*V2[1])*facteur
        V[2] = V[2] - (derivee1*V1[2] - derivee2*V2[2])*facteur
        
        print("La nouvelle vitesse initiale est")
        print(V)
        
        # print("checkpoint3")
        
        #Calcul de la nouvelle géodésique
        
        chemin = odeint(f1, depart + V, T)
        
        #Calcul du nouveau point d'arrivée
    
        fin = sp.Matrix([chemin[-1][0], chemin[-1][1], chemin[-1][2]])
    
        ecart = sp.sqrt(((fin - sp.Matrix(arrivee)).T * (fin - sp.Matrix(arrivee)))[0])
        
        iter = iter + 1
        
    
    return(chemin)



## Utilisation de l'algorithme
N = 1000
depart = [2,0,0]
arrivee = [0,2,0]
epsilon = 0.01

geodesique(depart,arrivee,N,epsilon)   


# X0 = [4, 0, 2, 2, 0.4, 0]

# chemin = odeint(f1, X0, T)

##

# def f2(X, t):
#     A = equa2.subs([(x, chemin[int(t * (N-1))][0]), (y, chemin[int(t * (N-1))][1]), (z, chemin[int(t * (N-1))][2]), (vx, chemin[int(t * (N-1))][3]), (vy, chemin[int(t * (N-1))][4]), (vz, chemin[int(t * (N-1))][5])])
#     A = A.subs([(px, X[0]), (py, X[1]), (pz, X[2]), (pvx, X[3]), (pvy, X[4]), (pvz, X[5])])
#     return([X[3], X[4], X[5], A[0], A[1], A[2]])
    
# variations1 = euler(f2, [0, 0, 0] + V1, T)
# variations2 = euler(f2, [0, 0, 0] + V2, T)


## Affichage
# 
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# 
# def affichage(chemin):
#     fig = plt.figure()
#     ax = Axes3D(fig)
# 
#         
#     ax.plot(chemin[:, 0], chemin[:, 1], chemin[:, 2], c='b', marker='.')
# 
#         
#     plt.show()
#     

## Fonctions
# 
# def f1(X, t):
#     """ fonction donnant (x'', y'', z'') a partir de (x, y, z, x', y', z') et de t """
#     A = equa1.subs([(x, X[0]), (y, X[1]), (z, X[2]), (vx, X[3]), (vy, X[4]), (vz, X[5])])
#     return([X[3], X[4], X[5], A[0], A[1], A[2]])
    
# def f2(X, t):
#     A = equa2.subs([(x, chemin[t][0]), (y, chemin[t][1]), (z, chemin[t][2]), (vx, chemin[t][3]), (vy, chemin[t][4]), (vz, chemin[t][5])])
#     A = A.subs([(px, X[0]), (py, X[1]), (pz, X[2]), (pvx, X[3]), (pvy, X[4]), (pvz, X[5])])
#     return([X[3], X[4], X[5], A[0], A[1], A[2]])



