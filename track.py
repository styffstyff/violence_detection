
import person
import json
import cv2
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from yoloseg import YOLOSeg
from yoloseg.utils import xyxy2xywh

def updateJson(p):

    if p.jsonFILENAME is None:
        print("vous n'avez pas définit de nom de fichier...")
        return
    else:
        filename = p.jsonFILENAME

    idImg = len(jsonData)



    value = {   "id": p.id, 
                "CentreTete": None, 
                "CentreCou": None,
                "BrasDroit": None,
                "longueurBrasDroit" : p.rightLen,
                "AngleBrasDroit": None,
                "BrasGauche": None, 
                "longeurBrasGauche" : p.leftLen,
                "AngleBrasGauche": None }

    if p.headCenter is not None:
        value["CentreTete"] = (int(p.headCenter[0]), int(p.headCenter[1]))

    if p.neckCenter is not None:
        value["CentreCou"] = (int(p.neckCenter[0]), int(p.neckCenter[1]))

    if p.right is not None:
        value["BrasDroit"] = (int(p.right[0]), int(p.right[1]))
    
    if p.thetaR is not None:
        value["AngleBrasDroit"] = p.thetaR * 180 / np.pi

    if p.left is not None:
        value["BrasGauche"] = (int(p.left[0]), int(p.left[1]))

    if p.thetaL is not None:
        value["AngleBrasGauche"] = p.thetaL * 180 / np.pi
    

    jsonData[idImg] = value

    return None

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
            try:
                distances.append(np.sqrt((x1-x2)**2+(y1-y2)**2))
            except:
                print("erreur sur le calcul de distance")
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
# backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=760, detectShadows=False)
# backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=1500, detectShadows=False)

model_path = "./YOLOv5_Segmentation/models/yolov5s-seg.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.3, iou_thres=0.3)

filename = "test1_SimpleMouvements" # Nom du fichier json
jsonData = {}

# cap=cv2.VideoCapture(0)
cap =cv2.VideoCapture("./DATA/video/mouvement_simple.mkv")
# surface = 10000
# cap=cv2.VideoCapture("Fight_OneManDown.mpg")
# cap=cv2.VideoCapture("Pedestrian.mp4") # 7 FPS
cap=cv2.VideoCapture("./DATA/video/Pedestrian.mp4") # 7 FPS
# cap = cv2.VideoCapture("Sample.MP4")
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

    if frame is None:
        break
    
    boxes, scores, class_ids, masks = yoloseg(frame)

    combined_img = yoloseg.draw_masks(frame) # Afficher le resultat de yolo

    

    boxes = xyxy2xywh(boxes)

    for i, box in enumerate(boxes):

        if class_ids[i] == 0 and len(box) > 0:
            x,y,w,h = box.astype(int)
            detected = person.Person((x, y, w, h), frame, masks[i])
            detected.setHBox()
            detected.setJsonFilename(filename)
            detected.autoSetHead()
            detected.autoSetNeck()
            detected.autoSetArms()
            currentPersonDict[detected]=True

    # On fait les prédictions:
    for i, kf in kFilterDict.items():
        kFilterDict[i].prediction() # on fait la prediction du filtre
        predict = kf.prediction() # on récupère la prédiction
        predictionDict[i] = predict # on ajoute la prédiction au dictionnaire
        px, py = int(predict[0][0]), int(predict[1][0]) 
        # cv2.arrowedLine(frame, (px,py), (px+int(predict[4]), py + int(predict[5])), (25,50,100), 2)

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
            """"
            predicted = person.Person((x, y, w, h), frame, fgMask)
            predicted.setHBox()
            predicted.setHeadNeckAndArms()
            predicted.setId(i)
            kFilterDict[i].update(np.array([[hx], [hy], [w], [h]]))
            personDict[i] = predicted
            """
                  
    for i, nb in compteur.items():
        if nb > 10:
            if i in list(kFilterDict.keys()):
                predictionDict.pop(i)
                kFilterDict.pop(i)

   
    for i, p in personDict.items():
        updateJson(p)
        p.displayBox(color = (23, 255, 0), nbBox=0)

    
    idimg += 1
    cv2.imshow("Detected Objects", combined_img) # YOLOv5 output
    cv2.imshow("frame", frame) # Boxes frame
    key=cv2.waitKey(70)&0xFF
    if key==ord('r'):
        rectangle = not rectangle
    if key==ord('t'):
        trace = not trace
    if key==ord('q'):
        quit()







filepath = "./DATA/JSON/test1_SimpleMouvements.json"

with open(filepath, "w") as f:
    json.dump(jsonData, f)



