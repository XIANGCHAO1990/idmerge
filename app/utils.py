# -*- coding: utf-8 -*-
import cv2
from PIL import Image as PImage, Image
from PIL import ImageFont, ImageDraw
# from ..config import BASE_DIR
import re
import base64
from io import BytesIO
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# BASE_DIR = '/Users/jingju/Desktop/workspace/Identy/identy'

def compressimage(img):
    width = img.width
    height = img.height
    rate = 1.0

    if width >= 2000 or height >= 2000:
        rate = 0.05
    elif width >= 1000 or height >= 1000:
        rate = 0.15
    elif width >= 500 or height >= 500:
        rate = 0.9

    width = int(width * rate)
    height = int(height * rate)

    img.thumbnail((width, height), Image.ANTIALIAS)
    return img


def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = PImage.open(image_data)

    return img

def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")  # Enregistre l'image dans le buffer
    myimage = buffer.getvalue()
    base64_str = base64.b64encode(myimage)
    base64_str = base64_str.decode('utf-8')
    return base64_str

def changeBackground(img, img_back, zoom_size, center):
    # 缩放
    img = cv2.resize(img, zoom_size)
    rows, cols, channels = img.shape

    # 转换hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 获取mask
    #lower_blue = np.array([78, 43, 46])
    #upper_blue = np.array([110, 255, 255])
    diff = [5, 30, 30]
    gb = hsv[0, 0]
    lower_blue = np.array(gb - diff)
    upper_blue = np.array(gb + diff)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow('Mask', mask)

    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    # 粘贴
    for i in range(rows):
        for j in range(cols):
            if dilate[i, j] == 0:  # 0代表黑色的点
                img_back[center[0] + i, center[1] + j] = img[i, j]  # 此处替换颜色，为BGR通道

    return img_back

name_font = ImageFont.truetype(BASE_DIR+'/static/hei.ttf', 52)
other_font = ImageFont.truetype(BASE_DIR+'/static/hei.ttf', 50)
bdate_font = ImageFont.truetype(BASE_DIR+'/static/fzhei.ttf', 50)
id_font = ImageFont.truetype(BASE_DIR+'/static/ocrb10bt.ttf', 54)

def mergeup(name,sex,nation,bir,headImg,address,idCardNo):
    im = PImage.open(BASE_DIR+'/static/up.jpg')
    avatar = base64_to_image(headImg)
    year = bir.split('-')[0]
    mon = bir.split('-')[1]
    day = bir.split('-')[2]

    draw = ImageDraw.Draw(im)
    draw.text((318, 217), name, fill=(0, 0, 0), font=name_font)
    draw.text((318, 352), sex, fill=(0, 0, 0), font=other_font)
    draw.text((733, 352), nation, fill=(0, 0, 0), font=other_font)
    draw.text((318, 495), year, fill=(0, 0, 0), font=bdate_font)
    draw.text((665, 495), mon, fill=(0, 0, 0), font=bdate_font)
    draw.text((855, 495), day, fill=(0, 0, 0), font=bdate_font)

    start = 0
    loc = 630
    while start + 18 < len(address):
        draw.text((318, loc), address[start:start + 18], fill=(0, 0, 0), font=other_font)
        start += 18
        loc += 80

    draw.text((313, loc), address[start:], fill=(0, 0, 0), font=other_font)
    draw.text((610, 990), idCardNo, fill=(0, 0, 0), font=id_font)

    avatar = cv2.cvtColor(np.asarray(avatar), cv2.COLOR_RGBA2BGRA)
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
    im = changeBackground(avatar, im, (480, 670), (220, 1220))
    im = PImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
    im = compressimage(im)
    im.save(BASE_DIR + '/up.png')
    basestr = image_to_base64(im)
    return basestr


def mergedown(police,expiryDate):
    im = PImage.open(BASE_DIR+'/static/down.jpg')

    draw = ImageDraw.Draw(im)
    draw.text((750, 852), police, fill=(0, 0, 0), font=other_font)
    draw.text((750, 992), expiryDate, fill=(0, 0, 0), font=other_font)
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
    im = PImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
    im = compressimage(im)
    im.save(BASE_DIR+'/down.png')
    basestr = image_to_base64(im)
    return basestr




