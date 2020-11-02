import traceback

import argparse
import math
import numpy as np
import os
import re
import tkinter as tk
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageTk


class Application(tk.Frame):
    def __init__(self, master, x, y, strokes, orig_text, extracted_string):
        super().__init__(master)
        self.master = master
        self.pack(side="top")
        x -= min(x)
        y -= min(y)

        self.x = x
        self.y = y
        self.strokes = strokes
        self.extracted_num = extracted_string
        self.resize_factor = 0.2

        self.initialise()

        self.text_label = tk.Label(master, text="Original text: " + orig_text)
        self.text_label.pack(side="top")
        self.extr_num_label = tk.Label(master, text="Extracted string: " + extracted_string)
        self.extr_num_label.pack(side="top")

        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack(side="bottom")

        self.focus_set()
        self.bind("<Key>", self.change_strokes)

        # Draw initial image
        self.draw_selected_strokes(strokes)

    def initialise(self):
        self.current_start = 0
        self.current_end = len(self.strokes)
        self.phase = 0
        self.width = int(math.ceil(max(self.x) * self.resize_factor))
        self.height = int(math.ceil(max(self.y) * self.resize_factor))
        self.tk_img = None

    def create_image_from_strokes(self, strokes):
        img = Image.new("RGB", (self.width, self.height), color=(255, 255, 255))
        img_canvas = ImageDraw.Draw(img)

        for stroke in strokes:
            for i in range(stroke[0], stroke[1] - 1):
                img_canvas.line((self.x[i] * self.resize_factor, self.y[i] * self.resize_factor,
                                 self.x[i + 1] * self.resize_factor, self.y[i + 1] * self.resize_factor),
                                fill=(0, 0, 0), width=3)
        return img

    def draw_selected_strokes(self, strokes):
        self.tk_img = ImageTk.PhotoImage(self.create_image_from_strokes(strokes))

        if self.canvas.find_withtag("img"):
            self.canvas.delete("img")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img, tags="img")

    def get_formatted_strokes(self, strokes):
        first = strokes[0][0]
        last = strokes[-1][1]

        combined = np.stack([self.x, -self.y], axis=1)[first:last]  # Inversion of y axis is necessary for actual synth
        combined_relative = combined[1:] - combined[:-1]
        combined_relative = np.insert(combined_relative, 0, [0., 0.], axis=0)  # Step above removes on element
        # normalisation to fit the numbers of the synth-lib
        combined_relative = np.divide(combined_relative, 20, dtype=np.float32)

        stroke_ends = [stroke[1] - strokes[0][0] - 1 for stroke in strokes]
        one_hot_stroke_ends = np.array([1. if i in stroke_ends else 0. for i in range(0, combined_relative.shape[0])],
                                       dtype=np.float32)

        formatted_strokes = np.stack([one_hot_stroke_ends, combined_relative[:, 0], combined_relative[:, 1]], axis=1)
        return formatted_strokes

    def quit(self):
        global formatted_strokes
        formatted_strokes = self.get_formatted_strokes(self.strokes[self.current_start:self.current_end])

        self.master.destroy()

    def change_strokes(self, event):
        # correct errors
        if event.keysym == "BackSpace":
            self.initialise()

        # remove sample from dataset
        if event.keysym == "r":
            global remove_sample
            remove_sample = True
            self.quit()
            return

        # change strokes
        if self.phase == 0:
            if event.keysym == "Right":
                self.current_start = min(self.current_start + 1, self.current_end)
            if event.keysym == "Left":
                self.current_start = max(self.current_start - 1, 0)
            if event.keysym == "Return":
                self.phase = 1
        elif self.phase == 1:
            if event.keysym == "Right":
                self.current_end = min(self.current_end + 1, len(self.strokes))
            if event.keysym == "Left":
                self.current_end = max(self.current_end - 1, self.current_start)
            if event.keysym == "Return":
                self.quit()
                return

        self.draw_selected_strokes(self.strokes[self.current_start:self.current_end])


def extract_relevant_lines(filename, regexes):
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
        all_matches = []
        for regex in regexes:
            match = re.findall("(" + regex + ")", line)
            all_matches.extend(match)

        if len(all_matches) > 1:
            for match in all_matches:
                relevant_lines.append((str(ln + 1).zfill(2), line, match))
        elif len(all_matches) == 1:
            relevant_lines.append((str(ln + 1).zfill(2), line, all_matches[0]))

    return relevant_lines


def process_file(in_filename, orig_text, extracted_string):
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

    root = tk.Tk()
    app = Application(root, x, y, strokes, orig_text, extracted_string)
    app.mainloop()

    global remove_sample
    if remove_sample:
        remove_sample = False
        return None

    global formatted_strokes
    return formatted_strokes


def write_results(descr_filename, stroke_filename, extracted_infos):
    extracted_infos = np.asarray(extracted_infos)
    dataset_description = extracted_infos[:, 1]
    dataset_description = [",".join(el) for el in dataset_description]
    stroke_list = extracted_infos[:, 0]

    with open(descr_filename, "w") as out:
        out.write("\n".join(dataset_description) + "\n")
    np.save(stroke_filename, np.asarray(stroke_list), allow_pickle=True)


def main():
    parser = argparse.ArgumentParser(
        description="A GUI tool for manually extracting strokes from online handwriting data.\n"
                    "Use the left and right key to define the beginning of the string and hit 'return' on completetion."
                    "Repeat the same for the end of the string. Hitting 'backspace' will reset the strokes and 'R' will"
                    "remove a sample from the dataset.\n"
                    "The format of the dataset should be the same as that of the IAMONDB dataset.")
    parser.add_argument("regex", nargs='+', help="The regexes for finding the strings to extract.")
    parser.add_argument("file_suffix", help="A suffix, e.g. 'num', that should be added to the filename of the results.")
    parser.add_argument("-ar", "--ascii-root-dir", default="ascii/",
                        help="The root directory where all the OCRs/CSRs for the strokes lie.")
    parser.add_argument("-sr", "--strokes-root-dir", default="lineStrokes/",
                        help="The root directory where all the stroke representations lie.")
    args = parser.parse_args()

    ascii_root_dir = args.ascii_root_dir
    strokes_root_dir = args.strokes_root_dir

    # we need dir + file name from the root dir to create xml filenames later
    ascii_file_list = [re.sub(ascii_root_dir, '', os.path.join(dp, f))
                       for dp, dn, fn in os.walk(ascii_root_dir) for f in fn]
    ascii_file_list = ascii_file_list[2:]

    extracted_infos = []
    for i, txt_filename in enumerate(ascii_file_list):
        full_txt_filename = os.path.join(ascii_root_dir, txt_filename)
        lines_with_info = extract_relevant_lines(full_txt_filename, args.regex)

        if len(lines_with_info) < 1:
            continue
        print("File", str(i) + "/" + str(len(ascii_file_list)))

        in_filename_wo_ext = os.path.splitext(txt_filename)[0]
        try:
            for line in lines_with_info:
                xml_filename = in_filename_wo_ext + "-" + line[0] + ".xml"
                full_xml_filename = os.path.join(strokes_root_dir, xml_filename)

                if not os.path.exists(full_xml_filename):
                    msg = "File skipped: " + full_xml_filename
                    print(msg)
                    with open("extraction.log", "a") as log_file:
                        log_file.write(msg + "\n")
                    continue

                orig_text = line[1]
                extracted_string = line[2]
                resulting_strokes = process_file(full_xml_filename, orig_text, extracted_string)

                if resulting_strokes is not None:
                    extracted_infos.append([resulting_strokes, [extracted_string, xml_filename, txt_filename]])

        except KeyboardInterrupt:
            with open("extraction.log", "a") as log_file:
                log_file.write("Interrupted at: " + txt_filename + "\n")
            break
        except Exception as err:
            print(traceback.format_exc())
            with open("extraction.log", "a") as log_file:
                log_file.write(txt_filename + " " + str(traceback.format_exc()) + "\n")

    result_filename = "iamondb_" + args.file_suffix
    write_results(result_filename + ".csv", result_filename + ".npy", extracted_infos)

    with open("extraction.log", "a") as log_file:
        log_file.write("------------------------------------------------------\n")


if __name__ == '__main__':
    global formatted_strokes
    global remove_sample
    remove_sample = False
    main()


