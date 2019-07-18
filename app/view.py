# -*- coding: utf-8 -*-

from flask import Flask
from flask import request, jsonify

# from identy.app.utils import base64_to_image, changeBackground
from .utils import mergeup, mergedown
from . import app
import os
import cv2
import numpy as np
from PIL import Image as PImage
from PIL import ImageFont, ImageDraw
import json
import re
import base64
from io import BytesIO
import logging

base_dir = os.path.dirname(os.path.dirname(__file__))


# def image_to_base64(image_path):
#     img = PImage.open(image_path)
#     output_buffer = BytesIO()
#     byte_data = output_buffer.getvalue()
#     base64_str = base64.b64encode(byte_data)
#     return base64_str

def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = PImage.open(image_data)

    return img

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


def paste(avatar, bg, zoom_size, center):
    avatar = cv2.resize(avatar, zoom_size)
    rows, cols, channels = avatar.shape
    for i in range(rows):
        for j in range(cols):
            bg[center[0] + i, center[1] + j] = avatar[i, j]
    return bg

@app.route('/', methods=['POST'])
def mergeidcard():
    if not request.data:
        return 'fail'

    cardsinfo = request.get_data(as_text=True)
    cardsinfo = json.loads(cardsinfo)
    name = cardsinfo['name']
    sex = cardsinfo['sex']
    nation = cardsinfo['nation']
    birthday = cardsinfo['birthday']
    police = cardsinfo['police']
    expiryDate = cardsinfo['expiryDate']
    address = cardsinfo['address']
    idCardNo = cardsinfo['idCardNo']
    headImg = cardsinfo['headImg']

    upbase = mergeup(name,sex,nation,birthday,headImg,address,idCardNo)
    downbase = mergedown(police,expiryDate)

    datas = {}
    datas['up'] = upbase
    datas['down'] = downbase
    datas = json.dumps(datas, indent=2)

    return jsonify(datas)


# @app.route('/', methods=['POST'])
# def mergeidcard():
#     if not request.data:
#         return 'fail'
#
#     datas = request.get_data(as_text=True)
#     cardsinfo = json.loads(datas)
#     name = cardsinfo['name']
#     sex = cardsinfo['sex']
#     nation = cardsinfo['nation']
#     birthday = cardsinfo['birthday']
#     police = cardsinfo['police']
#     expiryDate = cardsinfo['expiryDate']
#     address = cardsinfo['address']
#     idCardNo = cardsinfo['idCardNo']
#     headImg = cardsinfo['headImg']
#     year = birthday.split('-')[0]
#     mon = birthday.split('-')[1]
#     day = birthday.split('-')[2]
#
#     im = PImage.open(base_dir + '/static/empty.png')
#     avatar = base64_to_image(headImg)  # 500x670
#
#     name_font = ImageFont.truetype(os.path.join(base_dir + '/static/hei.ttf'), 72)
#     other_font = ImageFont.truetype(os.path.join(base_dir+'/static/hei.ttf'), 60)
#     bdate_font = ImageFont.truetype(os.path.join(base_dir+'/static/fzhei.ttf'), 60)
#     id_font = ImageFont.truetype(os.path.join(base_dir+'/static/ocrb10bt.ttf'), 72)
#
#     draw = ImageDraw.Draw(im)
#     draw.text((630, 690), name, fill=(0, 0, 0), font=name_font)
#     draw.text((630, 840), sex, fill=(0, 0, 0), font=other_font)
#     draw.text((1030, 840), nation, fill=(0, 0, 0), font=other_font)
#     draw.text((630, 980), year, fill=(0, 0, 0), font=bdate_font)
#     draw.text((950, 980), mon, fill=(0, 0, 0), font=bdate_font)
#     draw.text((1150, 980), day, fill=(0, 0, 0), font=bdate_font)
#     start = 0
#     loc = 1120
#     while start + 11 < len(address):
#         draw.text((630, loc), address[start:start + 11], fill=(0, 0, 0), font=other_font)
#         start += 11
#         loc += 100
#     draw.text((630, loc), address[start:], fill=(0, 0, 0), font=other_font)
#     draw.text((950, 1475), idCardNo, fill=(0, 0, 0), font=id_font)
#     draw.text((1050, 2750), police, fill=(0, 0, 0), font=other_font)
#     draw.text((1050, 2895), expiryDate, fill=(0, 0, 0), font=other_font)
#
#     avatar = cv2.cvtColor(np.asarray(avatar), cv2.COLOR_RGBA2BGRA)
#     im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
#     im = changeBackground(avatar, im, (500, 670), (690, 1500))
#     im = PImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
#     # im.save('color.png')
#     buffer = BytesIO()
#     im.save(buffer, format="PNG")  # Enregistre l'image dans le buffer
#     myimage = buffer.getvalue()
#     base64_str = base64.b64encode(myimage)
#
#     return base64_str



