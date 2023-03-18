import person
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from numpy import genfromtxt
import math
import os
import glob
import yolov5_master.detector as DETECT

def aspectRatioBox(boxList):
    # ratio hauteur largeur des boites
    # ToDo prendre en compte les dimmension de l'image
    ratioBoxList = []
    for (x,y,w,h) in boxList:
        if w < (1.2)*h:
            ratioBoxList.append((x,y,w,h))
    return ratioBoxList

def closest_person(predictionDict, personDict, minDist=120):
    """
    prediction: vecteur d'etat prédit d'une personne
    pDict: dictionnaire des personnes détectées 
    """
    
    pDict={}

    def distance(p1, listePoint):
        distances = []

        for p2 in listePoint:
            x1, y1, x2, y2 = *p1, *p2
            distances.append(np.sqrt((x1-x2)**2+(y1-y2)**2))
            
        return distances

    predictionId = []
    predictionList = []

    for i, prediction in predictionDict.items():
        predictionId.append(i)
        predictionList.append((int(prediction[0][0]), int(prediction[1][0])))

    tabPerson= []
    tabDist= []

    for p in personDict.keys():
        dist = distance(p.headCenter, predictionList)
        tabDist.append(dist)
        tabPerson.append(p)

    predictionId = np.array(predictionId)
    tabPerson = np.array(tabPerson)
    tabDist = np.array(tabDist)
    sortedDist = np.sort(tabDist, axis=None)

    for i, d in enumerate(sortedDist):
        if d>minDist:
            break
            
        id1, id2 = np.where(tabDist==d)

        if not len(id1) or not len(id2):
            continue
        
        # pour eviter de faire des calculs inutiles
        tabDist[id1, :]=minDist+1
        tabDist[:, id2]=minDist+1
        
        if len(predictionId)<1:
            continue
        else:
            for i, pid in enumerate(predictionId[id2]):
                pDict[int(pid)]=tabPerson[id1][i]
        
    return pDict


        



# methode de soustraction d'arrière plan
backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=760, detectShadows=False)



cap=cv2.VideoCapture("Pedestrian.mp4") # 7 FPS

surface = 500
personDict={} # clé: id de person valeur: info de person associés à id
kFilterDict={} # clé: id de person valeur: filtre de kalman
predictionDict={} # clé: id de person valeur: (prediction de Kalman, Estimation de l'erreur)
compteur={}
idP = 0
idimg = 0
dt = 0.142
#dt = 0.26 # devrait etre calculé en temps réel

while True:
    predit=[]
    currentPersonDict={}

    
    ret, frame = cap.read()
    cv2.imwrite(f'./images/test1{idimg}.jpg', frame)
    boxes = DETECT.run(source=f'./images/test1{idimg}.jpg', classes=0, view_img=False, nosave=True , device="cpu")

    if frame is None:
        break

    fgMask = backSub.apply(frame)

    fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1]
    
    for box in boxes:

        if len(box) > 0:
            x,y,w,h = box.astype(int)
            detected = person.Person((x, y, w, h), frame, fgMask)
            detected.setHBox()
            detected.setHeadNeckAndArms()
            currentPersonDict[detected]=True

    # On fait les prédictions:
    for i, kf in kFilterDict.items():
        kFilterDict[i].prediction() # on fait la prediction du filtre
        predict = kf.prediction() # on récupère la prédiction
        predictionDict[i] = predict # on ajoute la prédiction au dictionnaire
        px, py = int(predict[0][0]), int(predict[1][0]) 
        cv2.arrowedLine(frame, (px,py), (px+int(predict[4]), py + int(predict[5])), (25,50,100), 2)

    # On ajoute les premières personnes détectées 
    if len(personDict) == 0 and len(currentPersonDict) != 0:
        for p in currentPersonDict.keys():
            p.setId(idP)
            personDict[idP] = p
            kFilterDict[idP] = person.Track(p.getBox(), dt, p.headCenter)
            idP += 1
            
    
    else:
        # on récupère un dictionnaire qui donne toute les personnes détectée qui étaient présentes avant 
        nextPdict = closest_person(predictionDict, currentPersonDict)

        # AFFICHAGE DE DEBUGAGE
        # print(f" \n les détectés à l'image d'avant: { { i: p.headCenter for i,p in personDict.items()} } ")
        # print(f" \n les associables : {[ p.headCenter for p in list(currentPersonDict.keys())]} " )
        # print(f" \n les associés :  { {i: p.headCenter for i,p in nextPdict.items()} } ")
        # print(f" \n les prédictions : { {i: d[0:2] for i,d in predictionDict.items()} } ")
        
        # usr = None
        # while usr is None:
        #     usr = str(input("OK?"))

        # On met à jour toute les personnes qui ont été de nouveau détectée
        for i, p in nextPdict.items():
            compteur[i] = 0
            predictionDict.pop(i)
            if p in list(currentPersonDict.keys()):
                # on retire la personne des personnes détectées puis on met à jour le dico des personnes
                currentPersonDict.pop(p)
                p.setId(i)
                personDict[i] = p 
                kFilterDict[i].update(p.getMesure())
            else:
                print("error ")


        # On traite les personnes non détectées
        for p in currentPersonDict:
            p.setId(idP)
            personDict[idP] = p 
            kFilterDict[idP] = person.Track(p.getBox(), dt, p.headCenter)
            idP += 1
    
        for i, prediction in predictionDict.items():
            if i in list(compteur.keys()):
                compteur[i] += 1
            else:
                compteur[i] = 1
            

            hx, hy, w, h = tuple( int(param) for param in prediction[:4:])
            previousBox = personDict[i].getBox()
            if personDict[i].headCenter is None:
                continue
            dx, dy = tuple( np.array(personDict[i].headCenter) - np.array([hx, hy]))
            x, y = previousBox[0]-dx, previousBox[1]-dy
            
            predicted = person.Person((x, y, w, h), frame, fgMask)
            predicted.setHBox()
            predicted.setHeadNeckAndArms()
            predicted.setId(i)
            kFilterDict[i].update(np.array([[hx], [hy], [w], [h]]))
            personDict[i] = predicted
                  
    for i, nb in compteur.items():
        if nb > 10:
            if i in list(kFilterDict.keys()):
                predictionDict.pop(i)
                kFilterDict.pop(i)

   
    for i, p in personDict.items():
        p.displayBox(color = (23, 255, 0), nbBox=0)

    cv2.imshow('FG Mask', fgMask)
    cv2.imshow("frame", frame)
    
    idimg += 1
    key=cv2.waitKey(70)&0xFF
    if key==ord('r'):
        rectangle = not rectangle
    if key==ord('t'):
        trace = not trace
    if key==ord('q'):
        quit()
    

