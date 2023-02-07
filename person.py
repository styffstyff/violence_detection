import numpy as np
import cv2 
from random import randint
from scipy import ndimage
import matplotlib.pyplot as plt

# Objectif creer un objet person qui permette d'accéder
# Plus simplement à toutes les caractéristiques nécessaire et minimiser 
# les calculs

class Person():
    def __init__(self, box, frame, mask=None):
        self.box = box
        self.frame = frame
        self.mask = mask
        self.hBox = None
        self.headCenter= None
        self.neckCenter= None
        self.theta= None 
        self.id = None
    
    def __str__(self):
        return f"""hBox:{self.hBox} \nheadCenter:{self.headCenter} \nneckCenter:{self.neckCenter} \ntheta:{self.theta} \n"""

    def setHBox(self):
        x, y, w, h = self.box
        # On obtient les boites h1,h2,h3:
        self.hBox = divideBoundingBox(x, y, w, h)
    
    def displayBox(self, size=2, nbBox=3):
        boxList=self.hBox
        if nbBox == 3:
            x, y, w, h = self.box
            cv2.putText(self.frame,f"id:{self.getId()}",(x,int(y+3*h)),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,(0, 255, 255), 1)
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 0, 0), size)
        else:
            for box in boxList:
                x, y, w, h = box
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), size)
            cv2.putText(self.frame,f"id:{self.getId()}",(x,y+h),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,(0, 255, 255), 1)
        return None

    def setHeadNeckAndArms(self):
        H1 = self.getHbox(0)
        x,y,w,h = H1
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
        # print(hCroppedFrame)
        hFrame = cv2.inRange(hCroppedFrame, lowerb, upperb)
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
            cv2.imshow("cf", hCroppedFrame)
            
            self.headCenter = x+center0[1], y+center0[0]
            cv2.circle(self.frame, (self.headCenter), 2, (0, 255, 0), 2)
            
        if yprojection is None :
            return None
        
        # CALCULS POUR LE COU
        if self.headCenter is not None:
            # NOTE la valeur 1.18 est completement arbitraire et 
            # à modifier avec l'expérience
            sigma = -1
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
        for y in range(minY, maxY):
            if (self.frame[y][x] == np.array([255, 255, 255])).any() :
                return y
        return maxY
    
    def getId(self):
        return self.id 
    
    def getMesure(self):
        x, y = self.headCenter
        w, h = tuple(self.box[2::])
        return np.array([[x], [y], [w], [h]])


class Track():
    def __init__(self, box, dt):
        self.box = box
        self.dt = dt
        x, y, w, h = box 
        # Vecteur d'état
        # x, y, w. h, vx, vy, ax, ay .. je devrais peut-etre ajouter la qté d'acc
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
        v=1E-5

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
        self.P=np.matrix([[500, 0, 0, 0, 0, 0, 0, 0],
                        [0, 500, 0, 0, 0, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0, 0, 0],
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

if __name__ == "__main__":
    obj = Track((1, 34, 2, 5), 0.1)
    obj.prediction()
    obj.printing()
