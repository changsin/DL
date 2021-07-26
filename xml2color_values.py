#!/usr/bin/env python
#
# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
1. Load the input xml & get points of a polygon
2. Get all the points inside the polygon
3. Read the corresponding rgb values of the points
4. Take avg, min, max
5. Write the results into the output xml
"""

from __future__ import absolute_import, division, print_function

import os
from tkinter import *
# Tk for file selector
from tkinter import filedialog

import glog as log
import lxml.etree as etree
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.path import Path
from tqdm import tqdm

# Global Variables
XML_FILE = ""
XML_DIR = ""
IMG_DIR = "."


def calculate():
    if len(XML_FILE) > 1:
        calculate_color_values(XML_FILE, IMG_DIR)


def calculate_color_values(xml_file, img_dir):
    """
    calculate color values of all polygons and
    outputs the results in a separate xml
    """
    el_annotations = etree.Element('annotations')

    output_xml_filename = os.path.basename(os.path.splitext(xml_file)[0] + '.colors.xml')
    output_filename = os.path.join(IMG_DIR, output_xml_filename)

    root = etree.parse(xml_file).getroot()
    version_element = root.xpath('version')[0]
    meta_element = root.xpath('meta')[0]

    el_annotations.append(version_element)
    el_annotations.append(meta_element)

    log.info('Output file: [{}]'.format(output_filename))

    f = open(output_filename, 'w+')

    for image_element in tqdm(root.iter('image'), desc='Processing images from ' + xml_file):
        new_image_element = get_color_values_xml(image_element)
        el_annotations.append(new_image_element)

    output_xml = etree.tostring(el_annotations, encoding='unicode')

    f.write(str(output_xml))

    f.close()

    log.info('Saved the results to: [{}]'.format(output_filename))


def get_color_values_xml(image_element):
    """
    get color values of all polygons uner the image_element
    :return: returns the image_element with the added <color> sub-element
        which contains h,s,v channel values (avg, min, max)
    """
    def to_dict(element):
        """
        turns an xml element to a dictionary
        """
        element_dict = dict()
        for key, value in element.items():
            element_dict[key] = value

        return element_dict

    image_dict = to_dict(image_element)

    image_filename = get_image_file_name(image_dict['name'])
    log.info("Processing %s" % image_filename)

    image = load_image(image_filename)
    pixel_values = get_pixel_values(image)

    polygon_elements = [p for p in image_element.iter('polygon')]

    counter = 0
    plt.imshow(image)
    for polygon_element in polygon_elements:
        polygon_dict = to_dict(polygon_element)

        # parse the polygon points string to xs & ys
        xs, ys = to_points_arr(polygon_dict['points'])

        mask = get_surface_mask(xs, ys, image.width, image.height)

        mask_pixel_values = get_mask_pixel_values(pixel_values, mask)
        new_polygon_element = color_values2xml(mask_pixel_values, polygon_element)
        image_element.append(new_polygon_element)

        log.info("  Done with polygon {}".format(counter))
        counter += 1

        # plt.plot(xs, ys)
        # plt.scatter(mask[1], mask[0], c=mask_pixel_values/255.0)

    return image_element


def color_values2xml(color_values, polygon_element):
    """
    exports color_values to <color><h ... xml sub-element under the polygon_element
    :param color_values: color_values of the polygon
    :param polygon_element: the xml element
    :return: polygon_element with the sub color element
    """
    def create_channel_element(channel_name, channel_index):
        """
        creates a sub-element for the given channel
        :param channel_name: h, s, v
        :param channel_index: 0, 1, 2
        :return: <color><h avg...
        """
        element = etree.SubElement(color_element, channel_name)
        element.set('avg', "%.2f" % (averages[channel_index]))
        element.set('min', "%d" % (minimums[channel_index]))
        element.set('max', "%d" % (maximums[channel_index]))

        return element

    averages = np.average(color_values, axis=0)
    minimums = np.amin(color_values, axis=0)
    maximums = np.amax(color_values, axis=0)

    color_element = etree.SubElement(polygon_element, 'color')
    h_element = create_channel_element('h', 0)
    s_element = create_channel_element('s', 1)
    v_element = create_channel_element('v', 2)
    polygon_element.append(color_element)

    return polygon_element


def to_points_arr(points_str):
    """
    converts string representation of points array to numpy arrays
    """
    xs = list()
    ys = list()
    for token in points_str.split(";"):
        x, y = token.split(",")
        xs.append(float(x))
        ys.append(float(y))

    return np.array(xs), np.array(ys)


def get_pixel_values(image):
    """
    Get a numpy array of an image so that one can access values[x][y].
    https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python
    """
    image = image.convert("HSV")
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == "HSV":
        channels = 3
    elif image.mode == "L":
        channels = 1
    else:
        log.info("Unknown mode: {}".format(image.mode))
        return None

    return np.array(pixel_values).reshape((height, width, channels))


def get_image_file_name(image_name):
    """
    just get the image file name
    """
    return image_name[image_name.rfind("/") + 1:]


def load_image(image_filename):
    img_path = os.path.join(IMG_DIR, image_filename)
    return Image.open(img_path)


def get_surface_mask(xs, ys, width, height):
    """
    get the surface mask of the polygon
    :param xs: x points of a polygon
    :param ys: y points of a polygon
    :param width: width of the image
    :param height: height of the image
    :return: all points inside the polygon
    """

    # get the canvas of the image
    cys = np.array(np.arange(0, height, 1, dtype=np.float64))
    cxs = np.array(np.arange(0, width, 1, dtype=np.float64))

    # get the grid of booleans telling whether each pixel is inside the polygon or not
    grid = is_in_polygon(cxs, cys, xs, ys)
    # reshape to height & width
    mask = grid.reshape(height, width)
    # converts grid mask to points mask - coordinates of the masked area
    return np.where(mask)


def is_in_polygon(cxs, cys, xs, ys):
    """
    :param cxs: canvas xs
    :param cys: canvas ys
    :param xs: polygon x points
    :param ys: polygon y points
    :return: array of booleans telling whether that point is in the polygon or not
    """
    cxs = cxs.reshape(-1)
    cys = cys.reshape(-1)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    cxs, cys = np.meshgrid(cxs, cys)  # make a canvas with coordinates
    cxs, cys = cxs.flatten(), cys.flatten()
    canvas = np.vstack((cxs, cys)).T

    polygon = Path([(xs[i], ys[i]) for i in range(xs.shape[0])])

    return polygon.contains_points(canvas)


def get_mask_pixel_values(pixel_values, mask):
    """
    :param pixel_values: all pixel values
    :param mask: coordinates of the masked area
    :return: pixel values of the masked area of a polygon
    """
    color_values = list()
    for x, y in zip(mask[0], mask[1]):
        color_values.append(pixel_values[x, y])

    return np.array(color_values)


def main():
    # Tk for file selector
    window = Tk()
    window.title("BO/XML to HSV min/max/avg")
    window.geometry("800x500+100+100")
    window.resizable(False, False)

    global NAME_IMG_DIR
    NAME_IMG_DIR = IntVar()

    def XMLFile():
        global XML_FILE
        XML_FILE = filedialog.askopenfilename(initialdir="D:/",
                                              title="XML 파일 선택",
                                              filetypes=(("XML", "*.xml"), ("all files", "*.*")))
        labelXMLFile.config(text="XML:["+XML_FILE+"]")
        XML_DIR = ""

    def xmlDirectory():
        global XML_DIR
        XML_DIR = filedialog.askdirectory(initialdir="D:/", title="XML 경로 지정")
        labelXmlDir.config(text="XML DIR:["+XML_DIR+"]")
        XML_FILE = ""

    def imageDirectory():
        global IMG_DIR
        IMG_DIR = filedialog.askdirectory(initialdir="D:/", title="이미지 경로 지정")
        labeImgDir.config(text="Image DIR:["+IMG_DIR+"]")

    labelDesc1 = Label(window, text="Calculate HSV min/max/avg")
    labelDesc1.grid(row=1, column=0)
    labelDesc2 = Label(window, text="Procedure:{[[1] Select XML File -> [2]Select IMG Dir to SV/txt Name -> [3] Generate color values xml }")
    labelDesc2.grid(row=2, column=0)
    labelDesc3 = Label(window, text="  ")
    labelDesc3.grid(row=3, column=0)

    labelXMLFile = Label(window, text="XML File:[]")
    labelXMLFile.grid(row=4, column=0)
    btnXMLFile = Button(window, overrelief="solid", text="[1.1] Select XML File", width=30, command=XMLFile, repeatdelay=1000, repeatinterval=100)
    btnXMLFile.grid(row=5, column=0)

    labelXmlDir = Label(window, text="XML DIR:[]")
    labelXmlDir.grid(row=6, column=0)
    btnXmlDir = Button(window, overrelief="solid", text="[1.2] Select XML Directory", width=30, command=xmlDirectory, repeatdelay=1000, repeatinterval=100)
    btnXmlDir.grid(row=7, column=0)

    labeImgDir = Label(window, text="Image DIR:[]")
    labeImgDir.grid(row=8, column=0)
    btnImgDir = Button(window, overrelief="solid", text="[2] Select Image Directory", width=30, command=imageDirectory, repeatdelay=1000, repeatinterval=100)
    btnImgDir.grid(row=9, column=0)

    labelCvtMASK = Label(window, text="## Calculate color values ##")
    labelCvtMASK.grid(row=10, column=0)
    btnCvtMASK = Button(window, overrelief="solid", text="[3] Calculate color values", width=30, command=calculate, repeatdelay=1000, repeatinterval=100)
    btnCvtMASK.grid(row=11, column=0)
    
    window.mainloop()


if __name__ == "__main__":
    main()
