import json
import math
from copy import copy

import os
import xml.etree.ElementTree as ET

import numpy as np
import re
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk


def extract_relevant_lines(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    first_line = -1
    for i, line in enumerate(lines):
        if "CSR" in line:
            first_line = i + 2  # 2 lines after CSR, one blank in between
            break

    if first_line == -1:
        raise Exception("Seems like the file sucks.")

    relevant_lines = []
    for ln, line in enumerate(lines[first_line:]):
        matches = re.findall("(\\d+)", line)

        if len(matches) > 1:
            for match in matches:
                relevant_lines.append((str(ln + 1).zfill(2), line, match))
        elif len(matches) == 1:
            relevant_lines.append((str(ln + 1).zfill(2), line, matches[0]))

    return relevant_lines


class Application(tk.Frame):
    def __init__(self, master, x, y, strokes, orig_text, extracted_num):
        super().__init__(master)
        self.master = master
        self.pack(side="top")
        x -= min(x)
        y -= min(y)
        self.x = x
        self.y = y
        self.strokes = strokes
        self.current_start = 0
        self.current_end = len(strokes)
        self.phase = 0
        self.tk_img = None
        self.extracted_num = extracted_num

        self.text_label = tk.Label(master, text="Original text: " + orig_text)
        self.text_label.pack(side="top")
        self.extr_num_label = tk.Label(master, text="Extracted number: " + extracted_num)
        self.extr_num_label.pack(side="top")

        self.width = int(math.ceil(max(x)))
        self.height = int(math.ceil(max(y)))
        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack(side="bottom")

        self.focus_set()
        self.bind("<Key>", self.change_strokes)

        # Draw initial image
        self.draw_selected_strokes(strokes)

    def create_image_from_strokes(self, strokes):
        img = Image.new("L", (self.width, self.height), color=255)

        img_canvas = ImageDraw.Draw(img)

        for stroke in strokes:
            for i in range(stroke[0], stroke[1] - 1):
                img_canvas.line((self.x[i], self.y[i], self.x[i + 1], self.y[i + 1]))

        return img

    def draw_selected_strokes(self, strokes):
        self.tk_img = ImageTk.PhotoImage(self.create_image_from_strokes(strokes))

        if self.canvas.find_withtag("img"):
            self.canvas.delete("img")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img, tags="img")

    def get_formatted_strokes(self, strokes):
        first = strokes[0][0]
        last = strokes[-1][1]

        # TODO: normalisation? - sentence finden und vergleichen oder einfach alles selber machen
        combined = np.stack([self.x, self.y], axis=1)[first:last]
        combined_relative = combined[1:] - combined[:-1]
        combined_relative = np.insert(combined_relative, 0, [0., 0.], axis=0)

        stroke_ends = [stroke[1] - strokes[0][0] - 1 for stroke in strokes]
        one_hot_stroke_ends = np.array([1 if i in stroke_ends else 0 for i in range(0, combined_relative.shape[0])])

        formatted_strokes = np.stack([one_hot_stroke_ends, combined_relative[:, 0], combined_relative[:, 1]], axis=1)
        return formatted_strokes

    def quit(self):
        global formatted_strokes
        formatted_strokes = self.get_formatted_strokes(self.strokes[self.current_start:self.current_end])

        self.master.destroy()

    def change_strokes(self, event):
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
                self.quit()
                return
        self.draw_selected_strokes(self.strokes[self.current_start:self.current_end])


def process_file(in_filename, orig_text, extracted_num):
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

    root = tk.Tk()
    app = Application(root, x, y, strokes, orig_text, extracted_num)
    app.mainloop()


def write_results(descr_filename, stroke_filename, dataset_description, stroke_list):
    with open(descr_filename, "w") as out:
        out.write("\n".join(dataset_description))
    np.save(stroke_filename, np.asarray(stroke_list), allow_pickle=True)


def main():
    ascii_root_dir = "../../../rnnlib/examples/online_prediction/ascii"
    strokes_root_dir = "../../../rnnlib/examples/online_prediction/lineStrokes"

    ascii_file_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(ascii_root_dir) for f in fn]
    ascii_file_list = ["h06/h06-235/h06-235z.txt"]
    for txt_filename in ascii_file_list:
        full_txt_filename = os.path.join(ascii_root_dir, txt_filename)
        lines_with_info = extract_relevant_lines(full_txt_filename)

        if len(lines_with_info) < 1:
            continue

        in_filename_wo_ext = os.path.splitext(txt_filename)[0]
        dataset_description = []
        stroke_list = []
        for line in lines_with_info:
            xml_filename = os.path.join(strokes_root_dir, in_filename_wo_ext + "-" + line[0] + ".xml")

            if not os.path.exists(xml_filename):
                print(xml_filename)
                continue

            try:
                orig_text = line[1]
                extracted_num = line[2]
                process_file(xml_filename, orig_text, extracted_num)

                global formatted_strokes
                stroke_list.append(formatted_strokes)
                dataset_description.append(extracted_num)
            except Exception as err:
                with open("extraction.log", "a") as log_file:
                    log_file.write(xml_filename + " " + str(err) + "\n")

        write_results("../dataset_descriptions/iamondb_num.txt", "../datasets/iamondb_num.npy", dataset_description,
                      stroke_list)

        with open("extraction.log", "a") as log_file:
            log_file.write("------------------------------------------------------")

        # TODO: repeat option if I want to cut numbers like 1980 in 4?
        # TODO: invert y


if __name__ == '__main__':
    global formatted_strokes
    main()
