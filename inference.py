from screeninfo import get_monitors
import torch
import torch.nn as nn
import dlib
import cv2
import pyautogui
import sys
from backend.arguments import *
from backend.complexmodel import ComplexModel
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

args=sys.argv
keys=[("-m", "models/model.pth"), ('-b', "-.1")]
values = extract_values_with_defaults(keys=keys, argv=args)
model_loc=values["-m"]
bias=float(values["-b"])

def get_data():
    global data_
    landmarks_list = []
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        xes = []
        ys = []
        for j in range(68):
            x = landmarks.part(j).x
            xes.append(x)
            y = landmarks.part(j).y
            ys.append(y)
        landmarks_list.extend(xes)
        landmarks_list.extend(ys)
    # Remove the last item from the list
    landmarks_list = landmarks_list[:-1]
    return landmarks_list
def get_screen_dimensions():
    screen_dimensions = []
    i=0
    for monitor in get_monitors():
        i=i+1 
        screen_dimensions.append((monitor.width, monitor.height, monitor.x, monitor.y, i))
    return screen_dimensions
def main():
    global model_loc, bias
    # Define model parameters
    input_size = 135  # Example input size
    output_size = 1  # Example output size
    hidden_size = 10  # Example hidden size
    num_hidden_layers = 10  # Example number of hidden layers
    # Load the trained model
    model = ComplexModel(input_size, output_size, hidden_size, num_hidden_layers)
    model.load_state_dict(torch.load(model_loc))
    with torch.no_grad():
        p=0
        l=0
        while True:
            try:
                single_data = get_data()
                
                # Convert single line of data to tensor
                single_tensor = torch.tensor([single_data], dtype=torch.float32)

                # Make prediction
                prediction = model(single_tensor)
                l=prediction.tolist()[0][0]+bias
                print(l)
                #l=math.floor(l)
                l=round(l)
                
                d=get_screen_dimensions()
                
                if l>0 and l<len(d)+1:
                    width, height, x, y, i=d[l-1]
                    #print(f"{l} {p} {i}")
                    if p !=l:
                        pyautogui.moveTo(x+(width/2), (y+height/2))
                else:
                    pass
                    #print(l)
                p=l
                #print(prediction)
            except(RuntimeError):
                print("Runtime error")
            except (IndexError):
                print(f"Index error: {l}")
            except(KeyboardInterrupt):
                print("Interrupted with keyboard")
                break

if __name__ == "__main__":
    main()
