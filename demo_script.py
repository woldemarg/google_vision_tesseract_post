import cv2
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import pytesseract as pt
from pytesseract import Output
from shapely.geometry import Polygon
from demo_class import DemoOCR

# %%

pytrend = TrendReq()
kw_list = ['google vision', 'tesseract']
pytrend.build_payload(kw_list, timeframe='today 12-m')
ocr_trends = pytrend.interest_over_time()

hex_colors = ['#1b9e77', '#d95f02']

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)
ax.axes.yaxis.get_major_ticks()[0].label1.set_visible(False)
ax.axes.set_ylim((0, 110))
ax.set(ylabel='iterest over time (%)')
ax.yaxis.grid()
(ax
 .title
 .set_text("Google trends for search terms 'google vision' and 'tesseract'"))
ax.plot(ocr_trends['google vision'], color=hex_colors[0], label=kw_list[0])
ax.plot(ocr_trends['tesseract'], color=hex_colors[1], label=kw_list[1])
ax.legend(loc='lower left', frameon=False)

# %%

pt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
GGL_CRED = 'credentials/ocr_test.json'

# %%

FILE = 'demo_img/demo1.jpg'

with open(FILE, 'rb') as img_obj:
    img_bytes = img_obj.read()

# %%

demo1 = DemoOCR(GGL_CRED, img_bytes)


# %%

def get_tes_data(image_bytes):
    """Return recognized text with bounding boxes."""
    tes_rw = pt.image_to_data(demo1.bytes_to_image(image_bytes)[0],
                              output_type=Output.DICT)
    tes_df = pd.DataFrame.from_dict(tes_rw)
    tes_df['text'] = tes_df['text'].replace(r'^\s+$', '', regex=True)
    tes_fl = (tes_df[(tes_df['text'].astype(bool)) &
                     (tes_df['conf'] != 0)])
    tes_fl = tes_fl.apply(demo1.expand_coordinates, axis=1)
    tes_gm = tes_fl.apply(demo1.fit_to_poly, axis=1).reset_index(drop=True)
    return tes_gm


# %%

blurred = cv2.GaussianBlur(demo1.bytes_to_image(img_bytes)[0], (13, 13), 0)
blurred_bytes = cv2.imencode('.png', blurred)[1].tobytes()

# %%

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
titles = ['tesseract: original image', 'tesseract: blurred image']
images = [img_bytes, blurred_bytes]
for i in range(2):
    view = demo1.draw_rects(get_tes_data(images[i]), images[i], (228, 26, 28))
    ax[i].axis("off")
    ax[i].set_adjustable("box")
    ax[i].title.set_text(titles[i])
    ax[i].imshow(view)
fig.tight_layout()

# %%

tes_data_bl = get_tes_data(images[1])
view1 = demo1.draw_rects(tes_data_bl.loc[[12, 14, 17]],
                         blurred_bytes,
                         (228, 26, 28))
view1_bytes = cv2.imencode('.png', view1)[1].tobytes()
view2 = demo1.draw_rects(tes_data_bl.loc[[13, 15, 18]],
                         view1_bytes,
                         (0, 255, 28))

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(view2)

# %%

for g, p1 in tes_data_bl.iterrows():
    poly1 = Polygon(p1['poly'])
    overlapped = []
    for h, p2 in tes_data_bl.iterrows():
        if g == h:
            continue
        poly2 = Polygon(p2['poly'])
        if poly1.contains(poly2.centroid):
            overlapped.append(p2['poly'])
    if overlapped:
        leftmost_x0 = min([x[0][0] for x in overlapped])
        tes_data_bl.loc[g]['poly'][1] = (leftmost_x0 - 5,
                                         tes_data_bl.loc[g]['poly'][1][1])
        tes_data_bl.loc[g]['poly'][2] = (leftmost_x0 - 5,
                                         tes_data_bl.loc[g]['poly'][2][1])

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(demo1.draw_rects(tes_data_bl, blurred_bytes, (228, 26, 28)))

# %%

ggl_data = demo1.get_ggl_ocr_data(img_bytes)

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(demo1.draw_rects(ggl_data, img_bytes, (55, 126, 184)))

# %%

tes_data = demo1.get_tes_ocr_data(img_bytes)
ggl_data_orig, tes_data_orig = ggl_data.copy(), tes_data.copy()

SHOULD_RESTART = True
while SHOULD_RESTART:
    SHOULD_RESTART = False
    for i, trow in tes_data.iterrows():
        for j, grow in ggl_data.iterrows():
            tpoly = Polygon(trow['poly'])
            gpoly = Polygon(grow['poly'])
            if (demo1.has_intersection(tpoly, gpoly) and
                    not tpoly.equals(gpoly)):
                points = np.array(trow['poly'] + grow['poly'])
                min_x, min_y, max_x, max_y = demo1.get_bbox_values(points)
                bbox = [(min_x, min_y),
                        (max_x, min_y),
                        (max_x, max_y),
                        (min_x, max_y)]
                tes_data.at[i, 'poly'] = bbox
                ggl_data.at[j, 'poly'] = bbox
                SHOULD_RESTART = True
                break

collated = demo1.collate_data(ggl_data, tes_data, col_poly='poly')

baseline = collated.apply(demo1.compare_text, axis=1)
baseline = baseline[(baseline['res'] == 'matched') &
                    (baseline['text_g'].str.len() >= 2)]
baseline['init_scale'] = baseline.apply(demo1.get_scale_ratio, axis=1)

fig, ax = plt.subplots(1, 3, figsize=(15, 10))
titles = ['google vision data',
          'tesseract data',
          'merged areas with identical chars']
ocr_data = [ggl_data_orig, tes_data_orig, baseline]
rgb_colors = [(55, 126, 184), (228, 26, 28), (77, 175, 74)]
for i in range(3):
    view = demo1.draw_rects(ocr_data[i], img_bytes, rgb_colors[i])
    ax[i].axis("off")
    ax[i].set_adjustable("box")
    ax[i].title.set_text(titles[i])
    ax[i].imshow(view)
fig.tight_layout()

# %%

initial_bars = demo1.draw_stacked_bars(baseline,
                                       img_bytes,
                                       col_poly='poly')

ggl_legend = mpatches.Patch(color=hex_colors[0], label='google vision')
tes_legend = mpatches.Patch(color=hex_colors[1], label='tesseract')

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(initial_bars)
plt.legend(bbox_to_anchor=(0.5, -0.1),
           loc='lower center',
           handles=[ggl_legend, tes_legend],
           frameon=False)

# %%

noised = demo1.add_gaussian_noise(img_bytes, 0.23)
noised_bytes = cv2.imencode('.png', noised)[1].tobytes()

captcha = demo1.put_watermark(img_bytes, (180, 180, 180, 255))
captcha_bytes = cv2.imencode('.png', captcha)[1].tobytes()

fig, ax = plt.subplots(1, 3, figsize=(15, 10))
titles = ['blurred image',
          'noised image',
          'captcha-like']
view_bytes = [blurred_bytes, noised_bytes, captcha_bytes]
for i in range(3):
    iter_data = demo1.get_iter_data(view_bytes[i])
    iter_bars = demo1.draw_stacked_bars(iter_data,
                                        view_bytes[i],
                                        col_poly='poly_base')
    iter_bars_bytes = cv2.imencode('.png', iter_bars)[1].tobytes()
    iter_view = demo1.draw_rects(baseline, iter_bars_bytes, rgb_colors[2])
    ax[i].axis("off")
    ax[i].set_adjustable("box")
    ax[i].title.set_text(titles[i])
    ax[i].imshow(iter_view)
    if i == 0:
        ax[i].legend(bbox_to_anchor=(0.5, -0.1),
                     loc='lower center',
                     handles=[ggl_legend, tes_legend],
                     frameon=False)
fig.tight_layout()
