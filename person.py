import numpy as np
import cv2 
from random import randint
from scipy import ndimage
import matplotlib.pyplot as plt

# Objectif creer un objet person qui permette d'accéder
# Plus simplement à toutes les caractéristiques nécessaire et minimiser 
# les calculs

class Person(object):
    # TODO neckcenter     ✅
    # TODO armOrientation ✅
    # TODO headcenter     ✅
    def __init__(self, box, frame):
        self.box = box
        self.frame = frame
        self.hBox = None
        self.headCenter= None
        self.neckCenter= None
        self.theta= None 
    
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
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 0, 0), size)
        else:
            for box in boxList:
                x, y, w, h = box
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), size)
        return None

    def setHeadNeckAndArms(self):
        H1 = self.getHbox(0)
        x,y,w,h = H1
        croppedFrame = self.frame[y:y+h, x:x+w]
        hx,hy,hw,hh = x, y, w, int(h*0.65)
        hCroppedFrame = self.frame[hy:hy+hh, hx:hx+hw]
        xlen, ylen, _ =np.shape(croppedFrame)

        # On calcul la projection de y simultanément avec la mise 
        # en place de Z qui contient la couleur du pixel et sa position
        Z = []
        yprojection = np.zeros(ylen)
        color=0
        for idx in range(xlen):
            for idy in range(ylen):
                if (croppedFrame[idx][idy] == np.array([255, 255, 255])).any():
                    yprojection[idy]+=1
                    color = 1
                else:
                    color = 0
                # if  idx<hxlen and  idy<hylen:
                #     Z.append((color, idx, idy))

        # plt.plot(yprojection)
        # plt.waitforbuttonpress()
        # return None


        miny,maxy = None,None
        
        # CALCULS POUR LA TETE
        imgray = cv2.cvtColor(hCroppedFrame, cv2.COLOR_BGR2GRAY)
        contours,_= cv2.findContours(image = imgray, 
                                     mode = cv2.RETR_EXTERNAL, 
                                     method = cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        # On ne garde que le premier ordre de contour
        c_0=contours[0]
        # Calcul du moment 
        M = cv2.moments(c_0)

        lowerb = np.array([0, 0, 0], np.uint8)
        upperb = np.array([255, 255, 255], np.uint8)
        hFrame = cv2.inRange(hCroppedFrame, lowerb, upperb)
        blobs = hFrame > 100
        labels, nlabels = ndimage.label(blobs)
        # find the center of mass of each label
        t = ndimage.center_of_mass(hFrame, labels, np.arange(nlabels) + 1 )
        # calc sum of each label, this gives the number of pixels belonging to the blob
        s  =ndimage.sum(blobs, labels,  np.arange(nlabels) + 1 )
        cx, cy = t[s.argmax()]  # notation of output (y,x)
        
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])

        center0 = int(cx), int(cy)

        # Version k-means -> peut etre l'effectuer 2 fois une fois sur les x et une fois sur les y ou trouver un couplage unique x,y
        # Z = np.float32(Z)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # # On a seulement besoin de deux clusters 
        # # pour notre kmean ie blanc et noir
        # ret,label,center=cv2.kmeans(Z,1,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # center = np.uint8(center)
        # center0 = center[0]
        # for c in center:
            # print(f"center:{c} ,  y : {hylen}, y : {hxlen} ")
            
            # if 0 <= c[2] <= hylen and 0 <= c[1] <= hxlen:             # On conserve le seul centre qui est sur la silhouette blanche
            #     print(c)
            #     cv2.circle(hCroppedFrame, (c[2],c[1]), 1, (0, 255, 0), 1)
            #     cv2.imshow("cf", hCroppedFrame)
            #     if (croppedFrame[c[1]][c[2]] !=  np.array([0, 0, 0])).any():
            #         center0=c
        if center0 is not None:
            cv2.circle(hCroppedFrame, (center0[1],center0[0]), 2, (0, 255, 0), 2)
            cv2.imshow("cf", hCroppedFrame)
            
            self.headCenter = x+center0[1], y+center0[0]
            cv2.circle(hCroppedFrame, (self.headCenter), 2, (0, 255, 0), 2)
            
        
        # CALCULS POUR LE COU
        if self.headCenter is not None:
            # NOTE la valeur 1.18 est completement arbitraire et 
            # à modifier avec l'expérience
            sigma = -1
            # neckCenter = (self.headCenter[0]+ sigma*int(self.headCenter[0]*0.02), 400+ int(maxy))
            neckCenter = x+np.argmax(yprojection), int(self.headCenter[1]*1.03)
            # neckCenter = 27,30
            nx, ny = neckCenter
            print(nx, ny)
            cv2.circle(self.frame, (nx, ny), 2, (0,0,255), 2)
            self.neckCenter = neckCenter
        
        # CALCULS POUR L'ORIENTATION DES BRAS
        
        # déterminer de quel coté sont les bras
        gauche, droit = np.split(yprojection, [np.argmax(yprojection)])
        minDroit = np.argmin(droit)
        minGauche = len(gauche)-np.argmin(np.flip(gauche))
        print(f"minG: {minGauche}, mind: {minDroit}")
        miny = minDroit if abs((np.argmax(yprojection)-minDroit)) > abs((np.argmax(yprojection)-minGauche)) else minGauche
        miny = minDroit + x + len(gauche)
        miny2 = minGauche + x 
        print(miny)
        cv2.circle(self.frame, (miny2, y+h), 2, (255,0,255), 2)
        cv2.line(self.frame, (miny, y+h) , self.neckCenter, (0, 0, 255), 3) 
        if miny is not None and self.neckCenter is not None:
            theta = 0
            x2,y2=self.neckCenter
            if y2 != miny:
                theta =  np.arctan(x2/(abs(y2-miny)))
            self.theta = theta


    def getHbox(self, n):
        assert (n >= 0 and n < 3), "il n y'a que 3 boites"
        return self.hBox[n]


def divideBoundingBox(x, y, w, h):
    boxHeight = (h//3)
    boxes = ((x,y, w, boxHeight), (x,y+boxHeight, w, boxHeight), (x,y+(2*boxHeight), w, boxHeight))
    return boxes


def minMaxProjection(projection):
    """
    inutile
    """
    assert 1==0 , "STOoooooOP"
    def notNearPos(pos, dico):
        for i in [-2, -1, 1, 2]:
            if int(pos+i) in dico.keys():
                return False
        return True
    
    dx=1
    y=projection

    if sum(y) == 0:
        return None, None

    dy=np.gradient(y, dx)
    d2y=np.gradient(dy, dx)
    # On fait une approximation des zeros de la derivée en regardant quand elle change de signe
    # sgn = 1 signe + sgm = 0 signe -
    dyZeros = {}
    sgn = 1 if dy[0] > 0 else 0
    for val, pos in enumerate(dy):
        if (sgn == 1 and val < 0) or (sgn == 0 and val > 0) and notNearPos(pos, dyZeros):
            dyZeros[pos] = val
        sgn = 1 if val > 0 else 0
    # En isole deux extremum:
    minY = y[0],0              # Initialisation a améliorer peut-etre
    maxY = y[0],0
    for pos,val in dyZeros.items():
        pos=int(pos) # Vraiment ????
        if d2y[pos] < 0 and y[pos] > maxY[0]:
            maxY = y[pos],pos
        elif d2y[pos] >= 0 and y[pos] < minY[0]:
            minY = y[pos],pos
    
    return int(minY[0]),int(maxY[0])

if __name__ == "__main__":
    minMaxProjection([])