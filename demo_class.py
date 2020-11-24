from google.cloud import vision
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import cv2
import pytesseract as pt
from pytesseract import Output
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# %%

class DemoOCR():
    def __init__(self,
                 json_path,
                 img_bytes,
                 ranges=None,
                 rgb_colors=None):
        self.ranges = ([range(1, 30, 4),
                        range(1, 30, 4),
                        range(240, -30, -60)]
                       if ranges is None else ranges)
        self.rgb_colors = ([(27, 158, 119),
                            (217, 95, 2),
                            (117, 112, 179)]
                           if rgb_colors is None else rgb_colors)
        self.json_path = json_path
        self.img_bytes = img_bytes
        self.baseline = None
        self.iterations = None
        self.viz_data = None

    @staticmethod
    def bytes_to_image(image_bytes):
        img_BGR = cv2.imdecode(np.frombuffer(image_bytes, np.uint8),
                               cv2.IMREAD_COLOR)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        return img_RGB, img_BGR

    def add_gaussian_noise(self, image_bytes, s):
        orig_img = self.bytes_to_image(image_bytes)[0]
        copy_img = orig_img.copy()
        mean = (0, 0, 0)
        sigma = (s, s, s)
        cv2.randn(copy_img, mean, sigma)
        return orig_img + copy_img

    @staticmethod
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image,
                                rot_mat,
                                image.shape[1::-1],
                                flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def overlay_transparent(background, overlay):
        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0
        background[:, :] = ((1.0 - mask) *
                            background[:, :] +
                            mask *
                            overlay_image)
        return background

    def put_watermark(self,
                      image_bytes,
                      color=(250, 250, 250, 255),
                      mark='groupBWT    '):
        img = self.bytes_to_image(image_bytes)[0]
        size = img.shape[1] + img.shape[0]
        background_img = np.zeros((size, size, 4), dtype=np.uint8)
        mrgn = 20
        font_scale = 1.5
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = cv2.LINE_AA
        (w, h), base = cv2.getTextSize(mark, font, font_scale, thickness)
        line_height = h + base
        line = mark * round(size/w)
        n_lines = round(size / (line_height * 2))
        x, y0 = (mrgn, mrgn)
        for r in list(range(n_lines))[::2]:
            y = y0 + r * 2 * line_height
            cv2.putText(background_img,
                        line,
                        (x, y),
                        font,
                        font_scale,
                        color,
                        thickness,
                        line_type)
        backgroung_img_rot = self.rotate_image(background_img, 45)
        backgroung_img_rot2 = self.rotate_image(background_img, -45)
        background_grid = cv2.addWeighted(backgroung_img_rot,
                                          0.3,
                                          backgroung_img_rot2,
                                          0.7,
                                          0.0)
        dy = round(background_grid.shape[0] / 2 - img.shape[0] / 2)
        dx = round(background_grid.shape[1] / 2 - img.shape[1] / 2)
        backgroung_cropped = background_grid[dy: dy + img.shape[0],
                                             dx: dx + img.shape[1]]
        out = self.overlay_transparent(img, backgroung_cropped)
        return out

    def get_white_color_ratio(self, image_bytes):
        img = self.bytes_to_image(image_bytes)[1]
        white = [255, 255, 255]  # RGB
        sensivity = 20
        lower_white = np.array([white[2]-sensivity,
                                white[1]-sensivity,
                                white[0]-sensivity],
                               dtype=np.uint8)
        upper_white = np.array([white[2],
                                white[1],
                                white[0]],
                               dtype=np.uint8)
        white_mask = cv2.inRange(img, lower_white, upper_white)
        return (white_mask > 0).mean()

    @staticmethod
    def fit_to_poly(row):
        row['poly'] = [(row['x0'], row['y0']),
                       (row['x1'], row['y1']),
                       (row['x2'], row['y2']),
                       (row['x3'], row['y3'])]
        (row
         .drop(labels=['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
               inplace=True))
        return row

    @staticmethod
    def get_word_representation(doc):
        doc_data = []
        for page in doc.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_dict = {}
                        symbol_list = []
                        for symbol in word.symbols:
                            symbol_list.append(symbol.text)
                            word_dict['text'] = ''.join(symbol_list)
                        for b in range(4):
                            word_dict[''.join(['x', str(b)])] = (word
                                                                 .bounding_box
                                                                 .vertices[b]
                                                                 .x)
                            word_dict[''.join(['y', str(b)])] = (word
                                                                 .bounding_box
                                                                 .vertices[b]
                                                                 .y)
                        doc_data.append(word_dict)
        return doc_data

    def get_ggl_ocr_data(self, image_bytes):
        credentials = (service_account
                       .Credentials
                       .from_service_account_file(self.json_path))
        client = vision.ImageAnnotatorClient(credentials=credentials)
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
        document = response.full_text_annotation
        anno_ls = self.get_word_representation(document)
        anno_df = pd.DataFrame.from_dict(anno_ls)
        anno_gm = (anno_df
                   .apply(self.fit_to_poly, axis=1)
                   .reset_index(drop=True))
        return anno_gm

    @staticmethod
    def expand_coordinates(row):
        row = row.rename({'left': 'x0', 'top': 'y0'})
        row['x1'] = row['x0'] + row['width']
        row['y1'] = row['y0']
        row['x2'] = row['x1']
        row['y2'] = row['y0'] + row['height']
        row['x3'] = row['x0']
        row['y3'] = row['y2']
        return row

    def get_tes_ocr_data(self, image_bytes):
        img_arr = self.bytes_to_image(image_bytes)[1]
        anno_ls = pt.image_to_data(img_arr, output_type=Output.DICT)
        anno_df = pd.DataFrame.from_dict(anno_ls)
        anno_df['text'] = anno_df['text'].replace(r'^\s+$', '', regex=True)
        anno_df_fil = (anno_df[(anno_df['text'].astype(bool)) &
                               (anno_df['conf'] != 0)]
                       .copy())
        anno_df_fil = anno_df_fil.apply(self.expand_coordinates, axis=1)
        anno_gm = (anno_df_fil
                   .apply(self.fit_to_poly, axis=1)
                   .reset_index(drop=True))
        for g, p1 in anno_gm.iterrows():
            poly1 = Polygon(p1['poly'])
            overlapped = []
            for h, p2 in anno_gm.iterrows():
                if g == h:
                    continue
                poly2 = Polygon(p2['poly'])
                if poly1.contains(poly2.centroid):
                    overlapped.append(p2['poly'])
            if overlapped:
                leftmost_x0 = min([x[0][0] for x in overlapped])
                anno_gm.loc[g]['poly'][1] = (leftmost_x0 - 5,
                                             anno_gm.loc[g]['poly'][1][1])
                anno_gm.loc[g]['poly'][2] = (leftmost_x0 - 5,
                                             anno_gm.loc[g]['poly'][2][1])
        return anno_gm

    @staticmethod
    def convert_to_tuple_list(string):
        string_ls = string.strip('()').split('), (')
        tuple_ls = [tuple(map(int, x.split(', '))) for x in string_ls]
        return tuple_ls

    def collate_data(self, ggl_d, tes_d, col_poly):
        col_poly_str = col_poly + '_str'
        tes_d['ocr'] = 'tesseract'
        ggl_d['ocr'] = 'google'
        data = (pd
                .concat([tes_d, ggl_d],
                        ignore_index=True)
                .reset_index(drop=True))
        data[col_poly_str] = (data[col_poly]
                              .apply(lambda x: ', '.join(map(str, x))))
        data['text'] = (data
                        .groupby(['ocr', col_poly_str])['text']
                        .transform(''.join))
        data.drop_duplicates(subset=['ocr', col_poly_str],
                             inplace=True)
        data_tes = data[data['ocr'] == 'tesseract'].copy()
        data_ggl = data[data['ocr'] == 'google'].copy()
        data_reshaped = data_tes.merge(data_ggl,
                                       on=col_poly_str,
                                       how='outer',
                                       suffixes=['_t', '_g'])
        data_reshaped = data_reshaped[[col_poly_str, 'text_g', 'text_t']]
        data_reshaped['poly'] = (data_reshaped[col_poly_str]
                                 .apply(self.convert_to_tuple_list))
        data_reshaped.replace({np.nan: ''}, inplace=True)
        return data_reshaped

    def match_to_baseline_poly(self, row):
        for poly in self.baseline['poly']:
            if Polygon(poly).contains(Polygon(row['poly']).centroid):
                row['base_poly'] = poly
                break
        return row

    @staticmethod
    def filter_chars(row, col):
        a = ''.join(c for c in row[col] if c.isalnum())
        b = ''.join(c for c in row['text_g_base'] if c.isalnum())
        a_puncts = ''.join(p for p in row[col] if not p.isalnum())
        b_puncts = ''.join(p for p in row['text_g_base'] if not p.isalnum())
        p_pairs = zip(a_puncts, b_puncts)
        pairs = zip(a, b)
        correct = ''.join([c for c, d in pairs if c == d])
        p_correct = ''.join([c for c, d in p_pairs if c == d])
        return correct + p_correct

    def get_iter_data(self, image_bytes):
        base = (self.baseline if self.baseline is not None
                else self.get_baseline(self.img_bytes))
        ggl_iter = self.get_ggl_ocr_data(image_bytes)
        tes_iter = self.get_tes_ocr_data(image_bytes)[['text', 'poly']].copy()
        ggl_iter, tes_iter = [df.apply(self.match_to_baseline_poly,
                                       axis=1)
                              for df in [ggl_iter, tes_iter]]
        ggl_iter, tes_iter = [df[~df['base_poly'].isna()]
                              for df in [ggl_iter, tes_iter]]
        collate_iter = self.collate_data(ggl_iter,
                                         tes_iter,
                                         col_poly='base_poly')
        iter_data = collate_iter.merge(base,
                                       left_on='base_poly_str',
                                       right_on='poly_str',
                                       suffixes=('_new', '_base'))
        iter_data['text_g'] = iter_data.apply(self.filter_chars,
                                              args=('text_g_new',),
                                              axis=1)
        iter_data['text_t'] = iter_data.apply(self.filter_chars,
                                              args=('text_t_new',),
                                              axis=1)
        return iter_data

    @staticmethod
    def has_intersection(poly_a, poly_b):
        return (poly_a.contains(poly_b.centroid) or
                poly_b.contains(poly_a.centroid))

    @staticmethod
    def get_bbox_values(arr):
        minx = min(arr[:, 0])
        miny = min(arr[:, 1])
        maxx = max(arr[:, 0])
        maxy = max(arr[:, 1])
        return (minx, miny, maxx, maxy)

    @staticmethod
    def compare_text(row):
        if row['text_t'] == row['text_g']:
            row['res'] = 'matched'
        elif row['text_t'] == '':
            row['res'] = 'google'
        elif row['text_g'] == '':
            row['res'] = 'tesseract'
        else:
            row['res'] = 'mismatched'
        return row

    @staticmethod
    def get_scale_ratio(r):
        return round((r['poly'][1][0] - r['poly'][0][0]) / len(r['text_g']))

    def get_baseline(self, image_bytes):
        ggl = self.get_ggl_ocr_data(image_bytes)
        tes = self.get_tes_ocr_data(image_bytes)[['text', 'poly']].copy()
        should_restart = True
        while should_restart:
            should_restart = False
            for i, trow in tes.iterrows():
                for j, grow in ggl.iterrows():
                    tpoly = Polygon(trow['poly'])
                    gpoly = Polygon(grow['poly'])
                    if (self.has_intersection(tpoly, gpoly) and
                            not tpoly.equals(gpoly)):
                        points = np.array(trow['poly'] + grow['poly'])
                        min_x, min_y, max_x, max_y = (self
                                                      .get_bbox_values(points))
                        bbox = [(min_x, min_y),
                                (max_x, min_y),
                                (max_x, max_y),
                                (min_x, max_y)]
                        tes['poly'].iloc[i] = bbox
                        ggl['poly'].iloc[j] = bbox
                        should_restart = True
                        break
        collated = self.collate_data(ggl, tes, col_poly='poly')
        baseline = collated.apply(self.compare_text, axis=1)
        baseline = baseline[(baseline['res'] == 'matched') &
                            (baseline['text_g'].str.len() >= 2)]
        baseline['init_scale'] = baseline.apply(self.get_scale_ratio, axis=1)
        self.baseline = baseline
        return baseline

    @staticmethod
    def create_bar(x, y, w, h, color, img):
        sub_img = img[y: y + h, x: x + w]
        rect = np.zeros((h, w, 3), np.uint8)
        rect[:] = color
        res = cv2.addWeighted(sub_img, 0.4, rect, 0.6, 0.0)
        return res

    def draw_stacked_bars(self, df, image_bytes, col_poly):
        img = self.bytes_to_image(image_bytes)[1]
        for n, r in df.iterrows():
            x = r[col_poly][0][0]
            y = r[col_poly][0][1]
            h = (r[col_poly][3][1] - r[col_poly][0][1]) // 2
            gw = len(r['text_g']) * r['init_scale']
            tw = len(r['text_t']) * r['init_scale']
            if r['text_g'] != '' and r['text_t'] == '':
                img[y: y + h, x: x + gw] = self.create_bar(x,
                                                           y,
                                                           gw,
                                                           h,
                                                           self.rgb_colors[0],
                                                           img)
            elif r['text_g'] == '' and r['text_t'] != '':
                img[y + h: y + 2 * h, x: x + tw] = self.create_bar(x,
                                                                   y + h,
                                                                   tw,
                                                                   h,
                                                                   self.rgb_colors[1],
                                                                   img)
            else:
                img[y: y + h, x: x + gw] = self.create_bar(x,
                                                           y,
                                                           gw,
                                                           h,
                                                           self.rgb_colors[0],
                                                           img)
                img[y + h: y + 2 * h, x: x + tw] = self.create_bar(x,
                                                                   y + h,
                                                                   tw,
                                                                   h,
                                                                   self.rgb_colors[1],
                                                                   img)
        return img

    def do_iter(self, iter_range, fun, fun_type, image_bytes):
        iter_res = {}
        iter_viz = []
        for i in iter_range:
            try:
                if fun_type == 'blurring':
                    pic = fun(self.bytes_to_image(image_bytes)[0], (i, i), 0)
                elif fun_type == 'polluting':
                    white_color_ratio = self.get_white_color_ratio(image_bytes)
                    if 0 <= white_color_ratio <= 0.39:
                        r = 1.75
                    elif 0.4 <= white_color_ratio <= 0.69:
                        r = 75
                    else:
                        r = 100
                    pic = fun(image_bytes, i/r)
                elif fun_type == 'watermark':
                    pic = fun(image_bytes, (i, i, i, 255))
                ibts_n = cv2.imencode('.png', pic)[1].tobytes()
                iter_data = self.get_iter_data(ibts_n)
            except KeyError:
                continue
            else:
                iter_res[i] = {}
                iter_res[i]['g'] = iter_data['text_g'].str.len().sum()
                iter_res[i]['t'] = iter_data['text_t'].str.len().sum()
                viz = self.draw_stacked_bars(iter_data,
                                             ibts_n,
                                             col_poly='poly_base')
                iter_viz.append(viz)
        return (iter_res, iter_viz)

    def make_iterations(self):
        baseline = (self.baseline if self.baseline is not None
                    else self.get_baseline(self.img_bytes))
        max_q = baseline['text_g'].str.len().sum()
        iterations = [{} for x in range(3)]
        fun_names = [cv2.GaussianBlur,
                     self.add_gaussian_noise,
                     self.put_watermark]
        fun_types = ['blurring', 'polluting', 'watermark']
        for i in range(3):
            iterations[i]['res'], iterations[i]['viz'] = (self
                                                          .do_iter(self.ranges[i],
                                                                   fun_names[i],
                                                                   fun_types[i],
                                                                   self.img_bytes))
            flag = False if fun_types[i] == 'watermark' else True
            iterations[i]['df'] = (pd
                                   .DataFrame
                                   .from_dict(iterations[i]['res'])
                                   .T
                                   .sort_index(ascending=flag)
                                   .reset_index(drop=True)) / max_q * 100
        self.iterations = iterations
        return iterations

    def make_animation(self):
        d = (self.iterations if self.iterations is not None
             else self.make_iterations())
        for i, ls in enumerate(d):
            mask = ls['df']['t'] < ls['df']['t'].shift(-1)
            viz_to_del = []
            while any(mask):
                ls['df'] = ls['df'][~mask]
                viz_to_del.extend(list(mask[mask].index))
                mask = ls['df']['t'] < ls['df']['t'].shift(-1)
            d[i]['df'] = ls['df']
            for j in sorted(viz_to_del, reverse=True):
                del d[i]['viz'][j]
        frames = min([x['df'].shape[0] for x in d])
        for i, ls in enumerate(d):
            offset = len(ls['df']) - frames
            d[i]['df_sliced'] = ls['df'][offset:]
            d[i]['viz_sliced'] = ls['viz'][offset:]
        fig, ax = plt.subplots(2, len(d),
                               figsize=(15, 10),
                               sharey='row')
        fig.tight_layout()
        titles = ['Instant blurring', 'Noise pollution', 'Captcha mocking']
        lines = [[] for x in range(len(d))]
        plt.close(fig)
        for i in range(2):
            for j, r in enumerate(d):
                if i == 1:
                    ax[i, j].set_adjustable("box")
                    ax[i, j].spines["top"].set_visible(False)
                    ax[i, j].spines["bottom"].set_visible(False)
                    ax[i, j].spines["right"].set_visible(False)
                    ax[i, j].spines["left"].set_visible(False)
                    ax[i, j].set_yticks([0, 20, 40, 60, 80, 100], minor=True)
                    ax[i, j].set_yticks([0, 25, 45, 65, 85, 105])
                    ax[i, j].set_yticklabels([0, 20, 40, 60, 80, 100])
                    ax[i, j].yaxis.grid(True, which='minor')
                    ax[i, j].axes.set_ylim((0, 110))
                    (ax[i, j]
                     .axes
                     .yaxis
                     .get_major_ticks()[0]
                     .label1
                     .set_visible(False))
                    ax[i, j].axes.get_xaxis().set_visible(False)
                    ax[i, j].axes.set_xlim((0, frames - 1))
                    ax[i, j].tick_params(axis='y', which='both', length=0)
                    ax[i, j].tick_params(axis="y", direction="in", pad=-15)
                    if j == 0:
                        ax[i, j].set(ylabel='% of chars read per page')
                    line_g = (ax[i, j]
                              .plot([],
                                    [],
                                    color='#' + ''.join(f'{j:02X}' for j
                                                        in self.rgb_colors[0]))
                              [0])
                    line_t = (ax[i, j]
                              .plot([],
                                    [],
                                    color='#' + ''.join(f'{j:02X}' for j
                                                        in self.rgb_colors[1]))
                              [0])
                    lines[j].extend([line_g, line_t])

        def update(fr):
            for i in range(2):
                for j, r in enumerate(d):
                    if i == 0:
                        ax[i, j].clear()
                        ax[i, j].set_adjustable("box")
                        ax[i, j].title.set_text(titles[j])
                        ax[i, j].set_yticklabels([])
                        ax[i, j].set_xticklabels([])
                        ax[i, j].spines["top"].set_visible(False)
                        ax[i, j].spines["bottom"].set_visible(False)
                        ax[i, j].spines["right"].set_visible(False)
                        ax[i, j].spines["left"].set_visible(False)
                        ax[i, j].tick_params(axis='both',
                                             which='both',
                                             length=0)
                        if j == 0:
                            (ax[i, j]
                             .set(ylabel=r'$\sum$ of chars read per word'))
                        ax[i, j].set_adjustable("box")
                        ax[i, j].imshow(d[j]['viz_sliced'][fr])
                    else:
                        if j == 0:
                            ax[i, j].legend(['google vision', 'tesseract'],
                                            loc='lower left',
                                            frameon=False)
                        (lines[j][0]
                         .set_data(list(range(fr + 1)),
                                   d[j]['df_sliced'].iloc[:fr + 1]['g']
                                   .tolist()))
                        (lines[j][1]
                         .set_data(list(range(fr + 1)),
                                   d[j]['df_sliced'].iloc[:fr + 1]['t']
                                   .tolist()))
            return lines

        ani = animation.FuncAnimation(fig,
                                      update,
                                      frames=frames,
                                      interval=1250)
        self.viz_data = d
        return ani

    def draw_rects(self, df, image_bytes, color):
        img = self.bytes_to_image(image_bytes)[1]
        for n, r in df.iterrows():
            x0, y0, x2, y2 = (r['poly'][0][0], r['poly'][0][1],
                              r['poly'][2][0], r['poly'][2][1])
            img = cv2.rectangle(img, (x0, y0), (x2, y2), color, 2)
        return img
