# from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated
import glob
import json
import os
from modibot_metadata import ModibotMetadata
import numpy as np
from PIL import Image
import cv2
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated
from pprint import pprint


class ModiBot(RNGDataFlow):
    @staticmethod
    def display_image(inp, heatmap, as_numpy=False):

        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Image')
        plt.imshow(ModiBot.get_bgimg(inp))

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(ModiBot.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
        tmp = np.amax(heatmap, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig.clear()
            plt.close()
            return data

    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    def __init__(self, path, is_train=True):
        self.is_train = is_train
        self.path = path
        self.data = []
        self.idxs = []
        for i in range(1, 2):
            path = os.path.join(self.path, "pass_" + str(i))
            json_files = [f for f in glob.glob(os.path.join(path, "*.json"))]
            self.idxs += [os.path.join(path, os.path.basename(j).split(".")[0]) for j in json_files]
        pprint(self.idxs)

    def size(self):
        return len(self.idxs)

    def get_data(self):
        print("Getting data.")
        idxs = np.arange(self.size())
        if self.is_train:
            self.rng.shuffle(idxs)
        else:
            pass
        for cur in self.idxs:
            json_path = cur + ".json"
            img_url = cur + ".jpg"

            img_meta = {}
            img_meta['width'], img_meta['height'] = Image.open(img_url).size

            with open(json_path) as json_file:
                data = json.load(json_file)
                annotation = {}
                annotation['keypoints'] = data['bone_locations_2d']
                meta = ModibotMetadata(cur, img_url, img_meta, [annotation], sigma=8.0)
                yield [meta]