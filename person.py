import numpy as np
import json
import cv2 
import csv
from random import randint
from scipy import ndimage
import matplotlib.pyplot as plt

class Person():
    def __init__(self, box, frame, mask=None):
        self.box = box
        self.frame = frame
        self.mask = mask
        self.headCroppedCoord = None
        self.headCroppedMask = None
        self.hBox = None
        self.headCenter= None
        self.neckCenter= None
        self.right = None       # Bras "Droit"
        self.rightLen = None    # Longueur bras droit
        self.thetaR= None       # Angle associé au bras droit
        self.left = None        # Bras "Gauche"
        self.leftLen = None     # Longueur bras gauche
        self.thetaL= None       # Angle associé au bras gauche
        self.id = None
        self.jsonFILENAME = None
    
    def __str__(self):
        return f"""hBox:{self.hBox} \nheadCenter:{self.headCenter} \nneckCenter:{self.neckCenter} \nthetaL:{self.thetaL} \n"""

    def setJsonFilename(self, filename):
        self.jsonFILENAME = filename
    
    def setBox(self, box):
        self.box = box

    def setHBox(self):
        x, y, w, h = self.box
        # On obtient les boites h1,h2,h3:
        self.hBox = divideBoundingBox(x, y, w, h)
    
    def displayBox(self, size=2, color=(0, 255, 0), nbBox=3):
        boxList=self.hBox
        if nbBox == 3:
            x, y, w, h = self.box
            cv2.putText(self.frame,f"id:{self.getId()}",(x,int(y+3*h)),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,(0,255,255), 1)
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), color, size)
        else:
            for box in boxList:
                x, y, w, h = box
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), color, size)
            cv2.putText(self.frame,f"id:{self.getId()}",(x,y+h),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,(0,255,255), 1)
        return None
    
    def autoSetHead(self):
        if self.box is None or self.mask is None:
            return 
        else:
            H1 = self.getHbox(0)
            x, y, w, h = H1
            croppedMask = self.mask[y:y+h, x:x+w]
            yProj = np.array([ sum(croppedMask[:,i]) for i in range(len(croppedMask[0]))])

            # Je soustrait à la projection sur y la moyenne de cette projection pour n'obtenir que les 
            # "grandes valeurs" qui permettent d'identifier plus clairement les différentes "zones" et d'éviter
            # d'éventuels bruits parasites de la détéction. 
            m = np.mean(yProj, axis=0)

            yProj2 = yProj - m 
            length = {}
            tmp = None
            # Calcul des coordonnées des différentes "zones":
            for i, val in enumerate(yProj2):
                if val <= 0:    
                    yProj2[i] = 0

                if val <= 0 and tmp is not None:
                    length[(tmp, i)] = i-tmp
                    tmp = None
                else:
                    if tmp is None:
                        tmp = i

            # On identifie la plus grande zone, qui correspond approximativement à la zone de la tête
            if length:
                longest = max(length.values())
                x1, x2 = list((length.keys()))[list(length.values()).index(longest)]
            else:
                longest = None

            # On affine le masque de la tête
            if longest is not None:
                cx = int(x + min(x1, x2))
                marge = int(0.05*abs(x1-x2))
                if cx-marge > 0 and cx+abs(x1-x2)+marge < len(self.mask[0]):
                    cx -= marge

                    headCroppedMask = self.mask[y:y+h, cx: int(cx+abs(x1-x2)+2*marge)]
                    xProj = np.array([ sum(row) for row in headCroppedMask])
                    cy = len(xProj)//2 + np.argmin(xProj[len(xProj)//2::])               # On réduit le mask à la tête uniquement
                    headCroppedMask = self.mask[y:y+cy, cx: int(cx+abs(x1-x2)+2*marge)]  # ⏳ Pas super efficace etant donné qu'on redimmensionne deux fois le masque
                                                                                         # TODO réfléchir à une solution moins couteuse
                else:
                    headCroppedMask=self.mask[y:y+h, cx:cx+w] # Uniquement dans le cas où on n'arrive pas à obtenir un mask de la tête "précis"
    
                self.headCroppedCoord = (cx, y) # Je conserve le cou dans le masque pour les fonctions suivantes. 
                self.headCroppedMask = headCroppedMask
                
                hy, hx = np.array(ndimage.center_of_mass(headCroppedMask)).astype(int) # Calcul du centre de masse du masque de la tête -> centre de la tête

                # Afficher le mask en noir et blanc:
                # img = imagify(headCroppedMask) 
                # if img is not None:
                #     cv2.circle(img, (hx, hy), 4, (0, 255, 0), 4)
                #     cv2.imshow("head", img)
            
                # Afficher la projection des pixels:
                # h = np.arange(len(yProj))
                # plt.bar(h, yProj2)
                # plt.show()

                try:
                    self.headCenter = (cx+hx, y+hy)
                    cv2.circle(self.frame, self.headCenter, 2, (0, 255, 0), 2)
                except:
                    print("La tête est hors du champs")
    
    def autoSetNeck(self):
        if self.headCenter is None or self.mask is None or self.box is None or self.headCroppedCoord is None:
            return None
        else:
            H1 = self.getHbox(0)
            _, _, _, h = H1
            xh, yh = self.headCenter
            self.neckCenter = (xh, yh + int(0.33*h))
            return 
            if self.headCroppedMask is not None:
                x, y = self.headCroppedCoord
                yProj = np.array([ sum(self.headCroppedMask[:,i]) for i in range(len(self.headCroppedMask[0]))])    
                cx = x+np.argmax(yProj)
                xProj = np.array([ sum(row) for row in self.headCroppedMask[len(self.headCroppedMask)//2::] ]) # On voudrait eviter le minimum supérieur
                cy = y+len(self.headCroppedMask)//2+np.argmin(xProj)
                self.neckCenter = cx, cy
            else:
                H1 = self.getHbox(0)
                x, y, w, h = H1
                croppedMask = self.mask[y:y+h, x:x+w]
                yProj = np.array([ sum(croppedMask[:,i]) for i in range(len(croppedMask[0]))])    
                cx = x+np.argmax(yProj)
                xProj = np.array([ sum(row) for row in croppedMask[len(croppedMask)//2::] ]) # On voudrait eviter le minimum supérieur
                cy = y+len(croppedMask)//2+np.argmin(xProj)
                self.neckCenter = cx, cy
        
    def autoSetArms(self):
        if self.headCenter is None:
            return None
        else:
            x, y, w, h = self.box
    
            h = h//2
            croppedMask = self.mask[y:y+h, x:x+w] # Mask de la partie supérieure du corps
            
            n, p = np.shape(croppedMask)

            # Afficher le mask de yolo
            img = imagify(croppedMask)
            cv2.moveWindow("img", 40,30)

            xN, yN = self.neckCenter
            init = self.headCenter[0]-x, int(0.66 * h)-y  # point le plus proche de l'intersection du tronc et des bras
            Left, Right  = findTip(croppedMask, init) 
            # print(f"left arm: {Left}, right arm: {Right}")
            if Left is not None:
                xl, yl = Left
                left = x+xl, y+yl
                self.left = left
                self.leftLen = distance(left, self.neckCenter)

                if y + yl - yN != 0:
                    self.thetaL = np.arctan((xN - x + xl)/(y + yl - yN))
                else:
                    print("Même hauteur")
                    self.thetaL = np.pi/2

                if self.thetaL < 0:
                    self.thetaL = np.pi + self.thetaL
                print(f" angle à gauche:{self.thetaL * 180 / np.pi}")

                cv2.line(self.frame, self.neckCenter, left, (0, 0, 255), 3)
            
            if Right is not None:
                xr, yr = Right
                right = x+xr, y+yr
                self.right = right
                self.rightLen = distance(right, self.neckCenter)

                if y + yr - yN != 0:
                    self.thetaR = np.arctan((x+xr - xN) / (y+yr- yN))
                else:
                    print("Même hauteur")
                    self.thetaR = np.pi/2

                if self.thetaR < 0:
                    self.thetaR = np.pi + self.thetaR
                print(f" angle à droite:{self.thetaR * 180 / np.pi} \n")

                cv2.line(self.frame, self.neckCenter, right, (0, 0, 255), 3) 

            # Afficher le mask de yolo
            # img = imagify(croppedMask)
            # cv2.imshow("img", img) 
            
            
            # yProj = np.array([ sum(croppedMask[:,i]) if sum(croppedMask[:,i]) != 0 else len(croppedMask[0])+1   for i in range(len(croppedMask[0])) ])

            # #   Afficher la projection des pixels
            # z = np.arange(len(yProj))
            # plt.plot(z, yProj)
            # plt.show()

        cv2.circle(img, init, 10, (0, 0, 255))
        cv2.imshow("img", img) 
        return None


    def setHead(self, pos=None):
        if pos is None:
            self.autoSetHead()
        else:
            x, y = pos
            self.headCenter = x, y


    def setHeadNeckAndArms(self):
        return None
        # TODO modifier cette fonction pour qu'elle s'adapte au cas avec et sans YOLO
        H1 = self.getHbox(0)
        x,y,w,h = H1

        
        if len(self.mask.shape) == 2: # Avec le mask de YOLOseg
            croppedFrame = self.frame[y:y+h, x:x+w]

        if self.mask is None:
            croppedFrame = self.frame[y:y+h, x:x+w]
            croppedFrame2 = self.frame[y:y+(int(1.25*h)), x:x+w]
            hx,hy,hw,hh = x, y, w, int(h*0.65)
            hCroppedFrame = self.frame[hy:hy+hh, hx:hx+hw]
            xlen, ylen, _ =np.shape(croppedFrame2)
        else:
            croppedFrame2 = self.mask[y:y+(int(1.25*h)), x:x+w]
            hx,hy,hw,hh = x, y, w, int(h*0.65)
            hCroppedFrame = self.mask[hy:hy+hh, hx:hx+hw]
            xlen, ylen =np.shape(croppedFrame2)          
        # On calcul la projection du nb de pixel sur y   

        hxlen, hylen = np.shape(hCroppedFrame)

        yprojection = np.zeros(ylen)
        color=0
        for idx in range(xlen):
            for idy in range(ylen):
                if (croppedFrame2[idx][idy] == np.array([255, 255, 255])).any():
                    yprojection[idy]+=1
                    color = 1
                else:
                    color = 0

        miny,maxy = None,None
        
        # CALCULS POUR LA TETE
        lowerb = np.array(0, np.uint8)
        upperb = np.array(255, np.uint8)
        # lowerb = np.array([0, 0, 0], np.uint8)
        # upperb = np.array([255, 255, 255], np.uint8)
        
        # Bug lié aux prédictions sans détection
        # if np.shape(hCroppedFrame)[1] == 0 or np.shape(hCroppedFrame)[0] == 0:
        #     print("Error")
        #     return None
            
        if len(hCroppedFrame) < 1:
            print("error")
            return None
         
        try: 
            hFrame = cv2.inRange(hCroppedFrame, lowerb, upperb)
        except:
            print("error ??!!")
            return None
        blobs = hFrame > 100
        labels, nlabels = ndimage.label(blobs)
        # find the center of mass of each label
        t = ndimage.center_of_mass(hFrame, labels, np.arange(nlabels) + 1 )
        # calc sum of each label, this gives the number of pixels belonging to the blob
        s  =ndimage.sum(blobs, labels,  np.arange(nlabels) + 1 )
        cx, cy = t[s.argmax()]  # notation of output (y,x)
        center0 = int(cx), int(cy)
        if center0 is not None:
            cv2.circle(hCroppedFrame, (center0[1],center0[0]), 2, 100 , 2)
            # cv2.imshow("cf", hCroppedFrame) DEGUG
            
            self.headCenter = x+center0[1], y+center0[0]
            cv2.circle(self.frame, (self.headCenter), 2, (0, 255, 0), 2)
            
        if yprojection is None :
            return None
        
        # CALCULS POUR LE COU
        if self.headCenter is not None:
            # NOTE la valeur 1.18 est completement arbitraire et 
            # à modifier avec l'expérience, je pense qu'il faudrait utiliser une 
            # projection sur les x de maniere similaire à celle sur les y puis 
            # identifier le minimum pour obtenir la seconde coordonnée du cou
        
            # neckCenter = (self.headCenter[0]+ sigma*int(self.headCenter[0]*0.02), 400+ int(maxy))
            neckCenter = x+np.argmax(yprojection), int(self.headCenter[1]*1.03)
            # neckCenter = 27,30
            nx, ny = neckCenter

            cv2.circle(self.frame, (nx, ny), 2, (0,0,255), 2)
            self.neckCenter = neckCenter
        
        # CALCULS POUR L'ORIENTATION DES BRAS
        
        # déterminer de quel coté sont les bras
        
        gauche, droit = np.split(yprojection, [np.argmax(yprojection)])
        
        # TODO mettre ca au propre c'est le bazar !
        if len(gauche) == 0:
            return None


        minDroit = np.argmin(droit)
        minGauche = len(gauche)-np.argmin(np.flip(gauche))
        # print(f"minG: {minGauche}, mind: {minDroit}")
        miny = minDroit if abs((np.argmax(yprojection)-minDroit)) > abs((np.argmax(yprojection)-minGauche)) else minGauche
        miny = minDroit + x + len(gauche)
        y1 = self.getY(miny)
        x2 = minGauche + x 
        y2 = self.getY(x2)
        cv2.circle(self.frame, (x2, y2), 2, (255,0,255), 2)
        cv2.line(self.frame, (miny, y1) , self.neckCenter, (0, 0, 255), 3) 
        if miny is not None and self.neckCenter is not None:
            theta = 0
            x2,y2=self.neckCenter
            if y2 != miny:
                theta =  np.arctan(x2/(abs(y2-miny)))
            self.theta = theta
        
        return None

    def setId(self, id):
        self.id = id
        return None
    
    def getBox(self):
        return self.box

    def getHbox(self, n):
        assert (n >= 0 and n < 3), "il n y'a que 3 boites"
        return self.hBox[n]
    
    def getY(self, x):
        H1=self.getHbox(0)
        minY = H1[1]
        maxY = minY+int(1.50*H1[3])
        for y in range(minY, maxY-1):  #IndexError: index 210 is out of bounds for axis 0 with size 210
            try:
                if (self.frame[y][x] == np.array([255, 255, 255])).any() :
                    return y
            except IndexError:
                print("person.py ligne 319")
        return maxY
    
    def getId(self):
        return self.id 
    
    def getMesure(self):
        x, y = self.headCenter
        w, h = tuple(self.getBox()[2::])
        return np.array([[x], [y], [w], [h]])


class Track():
    # TODO 
    # implementer un calcul de l'estimation de l'erreur
    # Root Mean Squared Error (RMSE)
    #
    def __init__(self, box, dt, headCenter=None): 
        self.box = box
        self.dt = dt
        _, _, w, h = box 
        x, y = headCenter
        # Vecteur d'état
        # x, y, w. h, vx, vy, ax, ay .. je devrais ajouter la qté d'acc
        self.state = np.matrix([[x],[y],[w],[h],[0],[0],[0],[0]])
        # Matrice de transition
        self.transition = np.matrix([   [1, 0, 0, 0, self.dt, 0,       0.5*(self.dt)**2, 0],
                                        [0, 1, 0, 0, 0,       self.dt, 0,                0.5*(self.dt)**2],
                                        [0, 0, 1, 0, 0,       0,       0,                0],
                                        [0, 0, 0, 1, 0,       0,       0,                0],
                                        [0, 0, 0, 0, 1,       0,       0,                0],
                                        [0, 0, 0, 0, 0,       1,       0,                0],
                                        [0, 0, 0, 0, 0,       0,       1,                0],
                                        [0, 0, 0, 0, 0,       0,       0,                1]])
        # Matrice d'observation
        self.observation = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 0, 0]])

        # Matrices de bruit (supposé gaussien):

        # bruit relatif à l'évolution de l'objet
        v=1E-7

        self.objectNoise=np.matrix([[v, 0, 0, 0, 0, 0, 0, 0],
                                    [0, v, 0, 0, 0, 0, 0, 0],
                                    [0, 0, v, 0, 0, 0, 0, 0],
                                    [0, 0, 0, v, 0, 0, 0, 0],
                                    [0, 0, 0, 0, v, 0, 0, 0],
                                    [0, 0, 0, 0, 0, v, 0, 0],
                                    [0, 0, 0, 0, 0, 0, v, 0],
                                    [0, 0, 0, 0, 0, 0, 0, v]
                                    ])

        # bruit relatif aux caractéristiques de la caméra
        self.camNoise=np.matrix([[v, 0, 0, 0],
                                [0, v, 0, 0],
                                [0, 0, v, 0],
                                [0, 0, 0, v]])
        
        # covariance de l'erreur de la prediction
        self.P=np.matrix([[100, 0, 0, 0, 0, 0, 0, 0],
                        [0, 100, 0, 0, 0, 0, 0, 0],
                        [0, 0, 5, 0, 0, 0, 0, 0],
                        [0, 0, 0, 5, 0, 0, 0, 0],
                        [0, 0, 0, 0, 10000, 0, 0, 0],
                        [0, 0, 0, 0, 0, 10000, 0, 0],
                        [0, 0, 0, 0, 0, 0, 10000, 0],
                        [0, 0, 0, 0, 0, 0, 0, 10000]
                        ])
        
    def prediction(self):
        self.state = np.dot(self.transition, self.state)

        # calcul de la covariance de l'erreur
        self.P=np.dot(np.dot(self.transition, self.P), self.transition.T)+self.objectNoise
        return self.state
    
    def update(self, z):
        # z: np.array([[x], [y], [w], [h]])
        # Calcul du gain de Kalman
        S=np.dot(np.dot(self.observation, self.P), self.observation.T)+self.camNoise
        K=np.dot(np.dot(self.P, self.observation.T), np.linalg.inv(S))
        # Correction 
        ecart = z - np.dot(self.observation, self.state)
        self.state=np.round(self.state+np.dot(K, ecart))
        I=np.eye(self.observation.shape[1])
        self.P=(I-(K*self.observation))*self.P

        return self.state
    
    def printing(self):
        print(self.state)


def divideBoundingBox(x, y, w, h):
    boxHeight = (h//3)
    boxes = ((x,y, w, boxHeight), (x,y+boxHeight, w, boxHeight), (x,y+(2*boxHeight), w, boxHeight))
    return boxes

def imagify(tab):
    """
    j'utilise ceci pour transformer le mask que fournit yolo en une "image"
    """
    img = []
    for i in range(len(tab)):
        tmp = []
        for j in range(len(tab[0])):
            tmp.append(np.array([0, 0, 0]))
        if len(tmp) == 0:
            return None
        img.append(tmp)
    img = np.array(img)
    for i in range(len(tab)):
        for j in range(len(tab[0])):
            if tab[i][j] == 1:
                img[i][j] = [255,255,255]
            else:
                img[i][j] = [0,0,0]

    return img.astype(np.uint8)


def findBottom(tab):
    for i, val in enumerate(tab):
        if i+1 < len(tab) and val <= tab[i+1]:
            return i 
    return False


def isValid(mat, init, pos, value = 1, threshold = 0.50, AFFICHAGE=False ):
    """
    Permet de verifier que deux points appartiennent à une même "zone" de manière assez simple
    """

    if init[0] < pos[0]:
        return isValid(mat, pos, init)
    
    valCount = {value:0, "other": 0}

    maxX = len(mat[0])
    maxY = len(mat)
    x0, y0 = init
    x1, y1 = pos

    if x1 >= maxX or y1 >= maxY:
        return False

    if x1 == x0 or mat[y1][x1] != value:
        return False

    m = (y0 - y1) / (x0 - x1)

    for x in range(0, x0-x1):
        y = int(m * x + y1)
        if x+x1+1 < maxX and y+1 < maxY and mat[y][x+x1] == value:
            valCount[value] += 1
        else:
            valCount["other"] += 1
    
    percent = valCount[value] / sum(valCount.values())
    if AFFICHAGE:
        print(f"Il y'a {percent*100} pourcent de pixel correctes entre le tronc et l'extrémité testée")
    return percent >= threshold

def findTip(mat, init, value=1):
    """
    Cette fonction a pour but de trouver les extremités gauches et droites du corps
    [mat] est une liste de liste qui constitue un masque
    [init] est un tuple qui consitue la position de repère qui doit être le plus proche du haut du tronc
    """

    n, p = np.shape(mat)
    x0, y0 = init

    left, right = None, None

    if mat[y0][x0] != value:
        return left, right

    yr = 0
    yl = 0

    for xr in reversed(range(x0, p, 3)):
        yr = 0
        while yr < n and xr  < p and mat[yr][xr] != value:
            yr += 3
    
        if yr < n and isValid(mat, init, (xr, yr), AFFICHAGE=False):
            right = [xr, yr]
            break
    
    xr -= 15
    while yr+1 < n and mat[yr][xr] == value:
        yr += 1
        right[1] = yr
    
    if right is not None:
        right = tuple(right)
    
    for xl in range(0, x0, 1):
        yl = 0
        
        while yl < n and xl < p and mat[yl][xl] != value:
            yl += 1
        
        if yl < n and isValid(mat, init, (xl, yl), AFFICHAGE=False):
            left = [xl, yl]
            break
    
    xl += 15
    while yl+1 < n and mat[yr][xr] == value and isValid(mat, init, (xl, yl)):
        yl += 1
        left[1] = yl
    
    if left is not None:
        left = tuple(left)
    
    if left == right:
        return None, None

    return left, right

def distance(p1, p2):
    """
    distance euclidienne entre les points p1 et p2
    """
    if p1 is None or p2 is None:
        return None

    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2-x1)**2 + (y2- y1)**2)


if __name__ == "__main__":
    obj = Track((1, 34, 2, 5), 0.1)
    obj.prediction()
    obj.printing()