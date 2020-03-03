import math
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageTk
# from tkinter import *
import tkinter as tk

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def main():
    in_filename = "h06-235z-01.xml"

    tree = ET.parse(in_filename)
    root = tree.getroot()

    xml_strokes = root[1].findall("Stroke")

    x = []
    y = []
    i = 0
    strokes = []  # second part of the tuple is not part of the stroke but first index of next
    for stroke in xml_strokes:
        start = i
        for point in stroke.iter("Point"):
            x.append(int(point.attrib["x"]))
            y.append(int(point.attrib["y"]))
            i += 1
        end = i
        strokes.append((start, end))

    x = np.asarray(x)
    y = np.asarray(y)

    # Just some more hacky resizing
    x = x / 5
    y = y / 5

    x -= min(x)
    y -= min(y)

    root = tk.Tk()
    app = Application(root, x, y, strokes)
    app.mainloop()


class Application(tk.Frame):
    def __init__(self, master, x, y, strokes):
        super().__init__(master)
        self.master = master
        self.pack(side="top")
        self.x = x
        self.y = y
        self.strokes = strokes
        self.current_start = 0
        self.current_end = len(strokes)
        self.phase = 0
        self.tk_img = None

        self.width = int(math.ceil(max(x)))
        self.height = int(math.ceil(max(y)))
        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack(side="top")

        self.focus_set()
        self.bind("<Key>", self.change_strokes)

        # Draw initial image
        self.draw_selected_strokes(strokes)

    def draw_selected_strokes(self, strokes):
        img = Image.new("RGB", (self.width, self.height))

        img_canvas = ImageDraw.Draw(img)

        for stroke in strokes:
            for i in range(stroke[0], stroke[1] - 1):
                img_canvas.line((self.x[i], self.y[i], self.x[i + 1], self.y[i + 1]))
        self.tk_img = ImageTk.PhotoImage(img)

        if self.canvas.find_withtag("img"):
            self.canvas.delete("img")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img, tags="img")

    def change_strokes(self, event):
        print(event)
        if self.phase == 0:
            if event.keysym == "Right":
                self.current_start = min(self.current_start + 1, self.current_end)
            if event.keysym == "Left":
                self.current_start = max(self.current_start - 1, 0)
            if event.keysym == "Return":
                self.phase = 1
        else:
            if event.keysym == "Right":
                self.current_end = min(self.current_end + 1, len(self.strokes))
            if event.keysym == "Left":
                self.current_end = max(self.current_end - 1, self.current_start)
            if event.keysym == "Return":
                pass  # TODO: save and quit
        self.draw_selected_strokes(self.strokes[self.current_start:self.current_end])


if __name__ == '__main__':
    main()
