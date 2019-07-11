import os
import sys
from collections import defaultdict
from collections import namedtuple
from uuid import uuid1
import random
from random import choice, sample
from PIL import Image, ImageDraw, ImageFont

from multiprocessing import Pool

back_ground_dir = 'bg/'

font_data_dir = 'fonts/'

single_font_data_dir = 'single/'

multi_font_data_dir = 'mix/'

mac_os_file = '.DS_Store'

font_id_txt = 'font_id_dfp.txt'
bg_id_txt = 'bg_id_dfp.txt'

FontBgCombo = namedtuple("FontBgCombo", "fonts bg")


def get_all_font():
    font_files = os.listdir(font_data_dir)
    return [font_data_dir + x for x in font_files if x != mac_os_file]


def get_all_bg():
    bg_files = os.listdir(back_ground_dir)
    return [back_ground_dir + x for x in bg_files if x != mac_os_file]


def get_font_id_map():
    fonts = get_all_font()
    return {x: fonts[x] for x in range(len(fonts))}


def get_bg_id_map():
    bgs = get_all_bg()
    return {x: bgs[x] for x in range(len(bgs))}


def mk_single_data():
    fonts_map = get_font_id_map()
    bgs_map = get_bg_id_map()

    with open(font_id_txt, 'a') as f:
        for item in fonts_map.items():
            f.write("{} {}\n".format(item[0], item[1]))

    res = get_bg_font_combos(fonts_map, bgs_map, 3, 10)
    print(len(res))
    return res


def get_bg_font_combos(fonts_dict, bgs_dict, k, num_per_font=1000):
    font_set = set(fonts_dict.keys())
    bg_set = set(bgs_dict.keys())

    fonts_combo_nums = defaultdict(int)
    res = []
    while len(font_set) > k:
        font_ids = sample(font_set, k)
        bg_id = sample(bg_set, 1)[0]

        fonts = [[x, fonts_dict[x]] for x in font_ids]
        bg = [bg_id, bgs_dict[bg_id]]
        res.append(FontBgCombo(fonts, bg))
        for font_id in font_ids:
            fonts_combo_nums[font_id] += 1
            if fonts_combo_nums[font_id] >= num_per_font:
                font_set.remove(font_id)

    return res


def read_text():
    file_name = u'huanbei.txt'
    with open(file_name, 'r') as f:#, encoding='utf-8'
        texts = f.read().split(' ')

        start = random.randint(0, 80)
        num = random.randint(1, 4)
        s = texts[start]
        for j in range(num):
            s += texts[start + j]
            if (1 + j) % 2 == 0:
                s += '\n'
    return s


def draw_pic(font_bg_combo):
    fonts = font_bg_combo.fonts
    fonts_id = [x[0] for x in fonts]
    bg = font_bg_combo.bg
    bg_id = bg[0]
    id = str(uuid1())

    image = Image.open( bg[1])
    image = image.convert("RGB")
    y_ids = set(range(10, 20))
    x_ids = set(range(10, 40))
    pos_y = sample(y_ids, 1)[0]
    for font in fonts:
        # 字体设置
        font_size = set(range(10, 30))
        pos_x = sample(x_ids, 1)[0]
        font_size = sample(font_size, 1)[0]
        draw = ImageDraw.Draw(image)
        my_font = ImageFont.truetype(font[1], size=font_size)
        width = image.width
        height = image.height
        x = pos_x / 100
        y = pos_y / 100
        position = (int(width * x), int(height * y))
        draw.multiline_text(position, read_text(), font=my_font, fill=(255, 255, 255))
        pos_y+=20

    image.save((multi_font_data_dir + "{}_{}_{}.jpg").format(id, fonts_id, bg_id))#[x+109 for x in fonts_id]


if __name__ == '__main__':


    # len(fonts)=198,dfp:109
    res = mk_single_data()
    pool_num = 10
    p = Pool(pool_num)
    draw_pic(res[0])
    p.map(draw_pic, res)
