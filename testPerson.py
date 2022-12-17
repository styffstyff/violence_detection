import cv2
import person
import matplotlib.pyplot as plt

img = cv2.imread("Image_sample/test613.jpg")

for i in range(8):
    # r = cv2.selectROI("select the area", img)
    cv2.destroyAllWindows()
    # r = (490, 179, 50, 99)
    r = (350, 422, 67, 153)
    detected=person.Person(r, img)
    detected.setHBox()

    detected.setHeadNeckAndArms()
    detected.displayBox(nbBox=0)
    print(str(detected))

    cv2.imshow("img",img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()