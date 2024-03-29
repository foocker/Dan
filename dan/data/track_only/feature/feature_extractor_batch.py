import numpy as np
import cv2


class Extractor:
    def __init__(
            self,
            model_name,
            load_path,
            gpu_ids='0',
            use_gpu=True,
            height=256,
            width=128,
            seed=1,
            cls=0):
        self.model_name = model_name
        self.load_path = load_path
        self.gpu_ids = gpu_ids
        self.use_gpu = use_gpu
        self.height = height
        self.width = width
        self.seed = seed

        winSize = (20, 20)
        blockSize = (10, 10)
        blockStride = (5, 5)
        cellSize = (5, 5)
        nbins = 9
        self.hog = cv2.HOGDescriptor(
            winSize, blockSize, blockStride, cellSize, nbins)

    def __call__(self, input, batch_size=10, feature_type=0):
        """
        :param input: detected images, numpy array
        feature_type = 0 表示提取reid特征，1 表示提取hog特征
        :return: image features extracted from input
        """
        if feature_type == 1:
            winStride = (20, 20)
            padding = (0, 0)
            if isinstance(input, list):
                if len(input) == 0:
                    return np.array([])
                features = []
                for ind in input:
                    ind_ = cv2.resize(
                        ind, (100, 75), interpolation=cv2.INTER_LINEAR)
                    extracted_feature = self.hog.compute(
                        ind_, winStride, padding)
                    extracted_feature = extracted_feature.T
                    features.append(extracted_feature)
            else:
                input_ = cv2.resize(
                    input, (100, 75), interpolation=cv2.INTER_LINEAR)
                features = self.hog.compute(input_, winStride, padding)
                features = features.T
            features = np.vstack(features)
            return features
        else:
            # add dl model for multiclass
            return np.array([])

