import person
import numpy as np
import matplotlib.pyplot as plt
import cv2

def aspectRatioBox(boxList):
    # ratio hauteur largeur des boites
    # ToDo prendre en compte les dimmension de l'image
    ratioBoxList = []
    for (x,y,w,h) in boxList:
        if w < (1.2)*h:
            ratioBoxList.append((x,y,w,h))
    return ratioBoxList

def closest_person(p,personDict, keys=False ,minDist=100):
    """ 
    Prend un objet person p et une liste d'objets person personDict et renvoie 
    la personne la plus proche de p dans personDict avec une "sécurité" l'option
    minDist qui s'assure que si aucune des personnes n'est vraiment proche de p
    cette fonction ne renvoit rien 
    TODO modifier cette fonction pour utiliser la prédiction du filtre de Kalman
    """
    def rmse(mesure, prediction): 
        # Root mean squared error
        return np.sqrt(np.mean((mesure-prediction)**2))

    def distance(p1, p2):
        x1, y1, x2, y2 = *p1, *p2
        return np.sqrt((x1-x2)**2+(y1-y2)**2)
    
    currentClosest = None
    if keys is False:
        for person in personDict.values():

            if person.headCenter is not None:
                dist = distance(person.headCenter, p.headCenter)
                if dist < minDist:
                    minDist = dist
                    currentClosest = person
    else:
        for person in personDict.keys():
            if person.headCenter is not None:
                dist = distance(person.headCenter, p.headCenter)
                if dist < minDist:
                    minDist = dist
                    currentClosest = person

    return currentClosest



import cv2
import numpy as np
from numpy import genfromtxt
import math
import os
import glob

backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=760, detectShadows=False)


# cap=cv2.VideoCapture(0)
# surface = 30000
# cap=cv2.VideoCapture("Fight_OneManDown.mpg")
cap=cv2.VideoCapture("Pedestrian.mp4")
surface = 500
personDict={} # clé: id de person valeur: info de person associés à id
kFilterDict={} # clé: id de person valeur: filtre de kalman
predictionDict={} # clé: id de person valeur: (prediction de Kalman, Estimation de l'erreur)
idP = 0
dt = 0.26 # devrait etre calculé en temps réel

while True:
    currentPersonDict={}
    
    ret, frame = cap.read()

    #Detection de l'objet
    if frame is None:
        break
    frame = cv2.GaussianBlur(frame, (1, 5), 2)
    fgMask = backSub.apply(frame)
    fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1]
    frame_contour=frame.copy()
    contours, nada=cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Première boucle pour initaliser toutes les détections
    for c in contours:
        #On dessine les contours c
        #plus petit rectangle qui englobe le contour
        if cv2.contourArea(c)<surface or cv2.contourArea(c)>surface*4 :
             continue
        rect = cv2.minAreaRect(c)
        box = cv2.boundingRect(c)
        box = aspectRatioBox([box])
        if box:
            x,y,w,h=box[0]
            cv2.rectangle(frame_contour, (x, y), (x+w, y+h), (0, 0, 255), 6)
            detected = person.Person(box[0], frame, fgMask)
            detected.setHBox()
            detected.setHeadNeckAndArms()
            #detected.displayBox(nbBox=0)
            currentPersonDict[detected]=True

    # Trois configurations pour mettre a jour les personnes présentes
    
    if len(personDict) == 0 and len(currentPersonDict) != 0:
        for p in currentPersonDict.keys():
            p.setId(idP)
            idP += 1
            personDict[p.getId()] = p
            kFilterDict[p.getId()] = person.Track(p.getBox(), dt)
            predictionDict[p.getId()] = (kFilterDict[p.getId()].prediction(),-1)

            predict=kFilterDict[p.getId()].prediction()
            px, py = int(predict[0][0]), int(predict[1][0])
            cv2.circle(frame, (px,py), 3, (255, 0, 254), 3)
            cv2.arrowedLine(frame, (px,py), (px+int(predict[4]), py + int(predict[5])), (255,0,0), 2)

    elif len(personDict) <= len(currentPersonDict):
        for p in currentPersonDict.keys():
            #
            # TODO choisir entre la méthode du plus prés ou du plus prés avec la prédiction de Kalman
            #
            precP = closest_person(p, personDict)
            if precP is not None:
                personDict.pop(precP.getId())
                p.setId(precP.getId())
                personDict[precP.getId()] = p
                predictionDict[p.getId()] = (kFilterDict[p.getId()].prediction(),-1) # -1 valeur d'essai je n'ai pas encore calculé l'erreur
                
                predict=kFilterDict[p.getId()].prediction()
                print(predict)
                px, py = int(predict[0][0]),int(predict[1][0])
                cv2.circle(frame, (px,py), 3, (255, 0, 254), 3)
                cv2.arrowedLine(frame, (px,py), (px+int(predict[4]), py + int(predict[5])), (255,0,0), 2)

                kFilterDict[p.getId()].update(p.getMesure())
            else:
                p.setId(idP)
                personDict[idP] = p
                kFilterDict[p.getId()]=person.Track(p.getBox(), dt)
                predict = predictionDict[p.getId()] = (kFilterDict[p.getId()].prediction(),-1)
                
                predict=kFilterDict[p.getId()].prediction()
                print(predict)
                px, py = int(predict[0][0]),int(predict[1][0])
                print(px,py)
                cv2.circle(frame, (px, py), 3, (255, 0, 254), 3)
                cv2.arrowedLine(frame, (px,py), (px+int(predict[4]), py + int(predict[5])), (255,0,0), 2)
                kFilterDict[p.getId()].update(p.getMesure())
                idP+=1
    else:
        present={}
        notPresent=[]
        for p in personDict.values():
            # TODO idem cas 2
            nextP = closest_person(p, currentPersonDict, keys=True)
            
            if nextP is not None:
                currentPersonDict.pop(nextP)
                nextP.setId(p.getId())
                personDict[p.getId()] = nextP
                present[p.getId()] = True  
                predict = predictionDict[p.getId()] = (kFilterDict[p.getId()].prediction(),-1)  
                
                predict=kFilterDict[p.getId()].prediction()
                px, py = int(predict[0][0]),int(predict[1][0])
                print(predict)
                cv2.circle(frame, (px,py), 3, (255, 0, 254), 3)
                cv2.arrowedLine(frame, (px,py), (px+int(predict[4]), py + int(predict[5])), (255,0,0), 2)

                kFilterDict[p.getId()].update(p.getMesure()) 

            if p.getId() not in present.keys():
                notPresent.append(p.getId())
        for idNP in notPresent:
            kFilterDict.pop(idNP)
            personDict.pop(idNP)

    for p in personDict.values():
        break
        p.displayBox(nbBox=0)


    
    for i, kf in kFilterDict.items():
        break
        print((int(kf.state[0][0]), int(kf.state[1][0])))
        cv2.circle(frame, (int(kf.state[0][0]), int(kf.state[1][0])), 3, (255, 0, 254), 3)
            
    
    cv2.imshow('FG Mask', fgMask)
    # cv2.imwrite(f'./Image_sample/test{id}.jpg', fgMask)
    cv2.imshow('frame contour', frame_contour)
    cv2.imshow("frame", frame)
    key=cv2.waitKey(70)&0xFF
    if key==ord('r'):
        rectangle=not rectangle
    if key==ord('t'):
        trace=not trace
    if key==ord('q'):
        quit()
