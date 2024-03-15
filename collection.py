import csv
import tkinter as tk
from PIL import Image, ImageTk
from screeninfo import get_monitors
import dlib
import cv2
import sys
from backend.arguments import extract_values_with_defaults

args=sys.argv
keys=[('-o', "datasets/output.csv")]
values = extract_values_with_defaults(keys=keys, argv=args)
output=values['-o']

data_=[]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

keepgoing=True

def get_screen_dimensions():
    screen_dimensions = []
    i=0
    for monitor in get_monitors():
        i=i+1 
        screen_dimensions.append((monitor.width, monitor.height, monitor.x, monitor.y, i))
    return screen_dimensions

def get_center_position(window_width, window_height, screen_width, screen_height, xoff, yoff):
    x = (screen_width - window_width) // 2 + xoff
    y = (screen_height - window_height) // 2 + yoff
    return x, y
def add_data(data):
    global data_
    landmarks_list = []
    _, frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    _, _, _, _, _, i=data
    for face in faces:
        landmarks = predictor(gray, face)

        
        xes=[]
        ys=[]
        for j in range(68):
            x = landmarks.part(j).x
            xes.append(x)
            y = landmarks.part(j).y
            ys.append(y)
        landmarks_list.extend(xes)
        landmarks_list.extend(ys)
        landmarks_list.extend([i])
    if len(landmarks_list)>0:
        data_.append(landmarks_list)
def save_data(data):
    csv1_file_path = output
    with open(csv1_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = [f"X{i+1}" for i in range(68)]
        header.extend([f"Y{i+1}" for i in range(68)])
        header.extend(['D'])
        writer.writerow(header)

        writer.writerows(data_)
def stop():
    global keepgoing
    keepgoing=False
def cw(image_path, width, height, x, y, i, create_window_func):
    global keepgoing
    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{width}x{height}+0+0")  # Resize window to full screen
    root.attributes('-topmost', True)  # Make sure window stays on top

    # Focus the window on the display
    root.wm_focusmodel('active')
    root.after(1, lambda: root.focus_force())
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)

    label = tk.Label(root, image=photo)
    label.pack()

    window_width, window_height = image.size
    x, y = get_center_position(window_width, window_height, width, height, x, y)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    root.bind("<Escape>", lambda event: [root.destroy(), stop()])
    root.bind("<space>", lambda event: [root.destroy(), add_data((image_path, width, height, x, y, i))])

    root.mainloop()


def create_windows_on_all_monitors(image_path):
    screen_dimensions = get_screen_dimensions()

    for monitor in screen_dimensions:
        width, height, x, y, i = monitor
        cw(image_path, width, height, x, y, i, create_windows_on_all_monitors)


if __name__ == "__main__":
    while keepgoing:
        create_windows_on_all_monitors("eye-icon-sign-symbol-design-free-png.png")
    save_data(data_)