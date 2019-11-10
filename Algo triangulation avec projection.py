# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:02:14 2016

@author: theophanegregoir
"""
import numpy as np

class Point:
    def __init__(self, coordonnees, base, angle_front = 2*np.pi, angle_change = True, bord = False, indice = 0):
        self.coord = coordonnees
        self.base = base
        self.angle_front = angle_front
        self.angle_change = angle_change 
        self.bord = bord
        self.indice = indice
        

def surfacepoint(coord):
    grad = gradient(f, coord)
    coord_surface = coord - (f(coord)/np.inner(grad,grad))*grad
    n = grad/np.sqrt(np.inner(grad, grad))
    if n[0]>0.5 or n[1]>0.5:
        t1 = np.array([n[1], -n[0], 0])
        t1 = t1/np.sqrt(np.inner(t1,t1))
    else:
        t1 = np.array([-n[2], 0, n[1]])
        t1 = t1/np.sqrt(np.inner(t1,t1))
    t2 = np.cross(n,t1)
    base = np.array([[n[0],t1[0],t2[0]],[n[1],t1[1],t2[1]],[n[2],t1[2],t2[2]]])
    p = Point(coord_surface, base)
    return(p)
    
def calcul_angle_front(poly):
    n = len(poly)
    for i in range(n):
        p = points[poly[i]]
        if p.angle_change:
            v1 = points[poly[(i-1)%n]]
            v2 = points[poly[(i+1)%n]]
            # Calcul des coordonnées des voisins dans la base orthonormée associée au point p
            coord1 = np.dot(np.dot((np.linalg.inv(p.base)), (v1.coord - p.coord)), p.base)
            coord2 = np.dot(np.dot((np.linalg.inv(p.base)), (v2.coord - p.coord)), p.base)
            # On normalise les vecteurs pour le calcul de l'angle
            coord1_u = coord1/(np.linalg.norm(coord1))
            coord2_u = coord2/(np.linalg.norm(coord2))
            angle = np.arccos(np.clip(np.dot(coord1_u, coord2_u), -1.0, 1.0))
            p.angle_front = angle
            p.angle_change = False

def distance_check(liste_poly, delta):
    poly = liste_poly[0]
    n = len(poly)
    for i in range(n):
        if not(points[poly[i]].bord):
            for j in range(0, i-2):
                if not(points[poly[j]].bord):
                    if np.linalg.norm(points[poly[i]].coord - points[poly[j]].coord) < delta:
                        liste_poly[0] = poly[:j+1] + poly[i:]
                        new_poly = poly[j:i+1]
                        liste_poly.append(new_poly)
                        points[poly[i]].bord = True
                        points[poly[j]].bord = True
            for j in range(i+3, n):
                if not(points[poly[j]].bord):
                    if np.linalg.norm(points[poly[i]].coord - points[poly[j]].coord) < delta:
                        liste_poly[0] = poly[:i+1] + poly[j:]
                        new_poly = poly[i:j+1]
                        liste_poly.append(new_poly)
                        points[poly[j]].bord = True
                        points[poly[i]].bord = True
            for autre_poly in liste_poly[1:]:
                for j in range(len(autre_poly)):
                    if not(points[autre_poly[j]].bord):
                        if np.linalg.norm(points[poly[i]].coord - points[autre_poly[j]].coord) < delta:
                            points[poly[i]].bord = True
                            points[autre_poly[j]].bord = True
                            liste_poly[0] = poly[:i+1] + autre_poly[j:] + autre_poly[:j+1] + poly[i:]
                            liste_poly.remove(autre_poly)
                            # ATTENTION : il faut expand les premières occurences de p_i et p_j
                            # pour ne pas qu'ils apparaissent deux fois.
                            
def propagation(p, v1, v2):
    # Détermination du nombre nt de nouveaux triangles
    om = p.angle_front
    nt = np.floor((3 * om)/np.pi) + 1
    dom = om / nt
    # Traitements de cas extrêmes
    if dom < 0.8 and nt > 1:
        nt= nt - 1
        dom = om / nt
    if nt == 1 and dom > 0.8 and np.linalg.norm(v1.coord - v2.coord) > (1.25 * delta):
        nt = 2
        dom = dom / 2
    if om < 3 and (np.linalg.norm(p.coord - v1.coord) <= delta/2 or np.linalg.norm(p.coord - v2.coord) <= delta/2):
        nt = 1
    
    # Génération des nouveaux triangles
    if nt == 1:
        triangles.append((p.indice, v1.indice, v2.indice))
        return([])
    else:
        # On projette v1 orthogonalement sur le plan tangent à p
        coord1 = np.dot(np.dot((np.linalg.inv(p.base)), (v1.coord - p.coord)), p.base)
        proj1 = coord1[1:]
        # On calcule la matrice de rotation d'angle dom
        rot = np.array([[np.cos(dom), -np.sin(dom)], [np.sin(dom), np.cos(dom)]])
        # On crée les nouveaux points
        qi = proj1
        points_crees = []
        for i in range(1, nt):
            qi = np.dot(rot, qi)
            coordi = np.dot(np.dot(p.base, np.array([0, qi[0], qi[1]])), (np.linalg.inv(p.base)))
            pi = surfacepoint(coordi + p.coord)
            pi.indice = len(points)
            points.append(pi)
            points_crees.append(pi.indice)
        v1.angle_change = True
        v2.angle_change = True
        triangles.append((p.indice, v1.indice, points_crees[0]))
        for i in range(len(points_crees)):
            triangles.append((p.indice, points_crees[i-1], points_crees[i]))
        triangles.append((p.indice, v2.indice, points_crees[-1]))
        return(points_crees)
        
def triangulation(f, coord, delta):
    # Initialisation
    p0 = surfacepoint(coord)
    p0.angle_change = False
    points = [p0]
    for i in range(6):
        qi = p0.coord + delta * np.cos(i*np.pi/3) * p0.base[:][1]
        qi = qi + delta * np.sin(i*np.pi/3) * p0.base[:][2]
        pi = surfacepoint(qi)
        pi.indice = len(points)
        points.append(pi)
    poly = range(1, 7)
    liste_poly = [poly]
    triangles = []
    for i in range(6):
        triangles.append((poly[i-1], poly[i], 0))
    
    # Algorithme
        
                            
                
                    
    