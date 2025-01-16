import cv2
import torch
import numpy as np
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', source='local')
model.classes = [2]  #pouze auta (class ID 2 v COCO datasetu)


#Nastavení obrazu/kamery
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

track_single_car = True  #jedno nebo více aut

#Fullscreen
cv2.namedWindow("Detekce Aut", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Detekce Aut", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

#Velikost a umístění regionu
region_x = 500  #souřadnice x
region_y = 150  #souřadnice y
width_region = 950  #šířka rámečku
height_region = 800  #výška rámečku

#Minimální velikost rámečku
min_box_area = 2800  

#Proměnné pro sledování aktuálního auta
tracked_car = None
last_car_center = None
last_change_time = None
last_box_area = None
max_jump_distance = 135  #v pixelech
time_threshold = 0.4  #po jaké době nečinosti hledat auto


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # Zelený text
thickness = 2


#Je vzdálenost mezi centry aut větší než maximální povolený skok?
def is_large_jump(last_car_center, car_center_x, car_center_y, box_area, max_jump_distance_weight=0.81):
    if last_car_center is None:
        return 0
    
    #Dynamické upravování max_jump_distance na základě velikosti rámečku
    if box_area > 7000:
        max_jump_distance = 135  #Pro větší objekty
    else:
        max_jump_distance = 30   #Pro menší objekty
    
    last_car_center_x, last_car_center_y = last_car_center
    center_distance = np.sqrt((car_center_x - last_car_center_x) ** 2 +
                               (car_center_y - last_car_center_y) ** 2)
    
    jump_score = 1 if center_distance > max_jump_distance else 0

    return jump_score * max_jump_distance_weight


def has_position_stable(last_change_time):
    if last_change_time is None:
        return False
    current_time = time.time()
    return current_time - last_change_time >= time_threshold

def is_large_size_change(last_box_area, current_box_area, size_change_threshold=13500, size_change_weight=0.19):
    #Ověř, zda jsou obě hodnoty platné
    if last_box_area is None or current_box_area is None:
        return 0
    
    size_change = 1 if abs(current_box_area - last_box_area) > size_change_threshold else 0
    return size_change * size_change_weight

while True:
    ret, frame = cap.read()
    if not ret:
        break

    text_x = frame.shape[1] - 150
    text_y = 30

    #Detekce objektů
    results = model(frame)
    detections = results.pandas().xyxy[0]

    mode_text = "SINGLE VIEW" if track_single_car else "MULTI VIEW"

    #Rozměrů obrazu
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    #Prioritní region
    region_left = region_x
    region_right = region_x + width_region
    region_top = region_y
    region_bottom = region_y + height_region

    best_car = None
    best_weight = -1

    for _, row in detections.iterrows():
        if row['class'] == 2:  #chci pouze auta
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            car_center_x = (x1 + x2) // 2
            car_center_y = (y1 + y2) // 2
            box_area = (x2 - x1) * (y2 - y1)

            #Podmínky pro výběr auta:
            if (region_left <= car_center_x <= region_right and
                region_top <= car_center_y <= region_bottom and
                box_area >= min_box_area):

                #Váha na základě vzdálenosti od středu 
                distance_to_center = np.sqrt((car_center_x - center_x) ** 2 + (car_center_y - center_y) ** 2)
                center_weight = 1 / (distance_to_center + 1)  #Vyšší váha pro auta blíže středu

                #Váha na základě plochy
                area_weight = box_area / (width * height)

                weight = 0.9 * center_weight + 0.1 * area_weight

                if track_single_car: #Jedno auto

                    if weight > best_weight:
                        best_weight = weight
                        best_car = (x1, y1, x2, y2, car_center_x, car_center_y)

                else:  #Sleduj Všechna auta

                    if box_area >= min_box_area:  #
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = "Auto "+f"- Ram: {box_area}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #Pokud bylo nalezeno nové auto, kontrolujeme zda došlo k příliš velkému skoku
    if best_car:
        x1, y1, x2, y2, car_center_x, car_center_y = best_car
        box_area = (x2 - x1) * (y2 - y1)

        if tracked_car:
            #Kontrola, zda by došlo k příliš velkému skok
            
                jump_weight = is_large_jump(last_car_center, car_center_x, car_center_y, box_area, max_jump_distance_weight=0.81)  
                size_change_weight = is_large_size_change(last_box_area, box_area, size_change_threshold=13500, size_change_weight=0.19)  

                if jump_weight > size_change_weight:
                    if jump_weight > 0:

                        car_center_x, car_center_y = last_car_center
                else:
                #jdeme na jiné sledované auto
                    tracked_car = best_car
                    last_car_center = (car_center_x, car_center_y)
                    last_box_area = box_area
                    last_change_time = time.time()  #čas změny pozice
        else:
            #Pokud nemáme sledované auto, nastavíme aktuální
            tracked_car = best_car
            last_car_center = (car_center_x, car_center_y)
            last_change_time = time.time()  # Nastavíme čas na aktuální

    if tracked_car and has_position_stable(last_change_time):
        tracked_car = None  
        last_box_area = None
    #Pokud je sledované auto, vykreslíme ho
    if tracked_car:
        x1, y1, x2, y2, car_center_x, car_center_y = tracked_car

        box_width = x2 - x1
        box_height = y2 - y1
               
        #Vykreslení středu rámečku
        cv2.circle(frame, (car_center_x, car_center_y), radius=6, color=(0, 0, 255), thickness=-1)

        #Vykreslení auta a jeho středu
        label = "Auto "+f"- Ram: {box_width * box_height}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    #Zobrazení prioritního regionu 
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 0, 0), 2)

    #Zobrazení souřadnic červeného bodu v levém horním rohu
    if last_car_center:
        coord_text = f"Souradnice: {last_car_center}"
        cv2.putText(frame, coord_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        text_size = cv2.getTextSize(mode_text, font, font_scale, thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10  

    cv2.putText(frame, mode_text, (text_x, text_y), font, font_scale, font_color, thickness)

    # Zobrazení snímku
    cv2.imshow("Detekce Aut", frame)

    #Klávesové vstupy
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): #vypni aplikaci
        break
    elif key == ord('f'): #režim fullscreen
        if is_fullscreen:
            cv2.setWindowProperty("Detekce Aut", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty("Detekce Aut", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = not is_fullscreen
    elif key == ord('c'):  #přepínání sledovacího režimu
        track_single_car = not track_single_car

# Uvolnění zdrojů
cap.release()
cv2.destroyAllWindows()