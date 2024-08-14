import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle

class Dataset(object):

    def __init__(self, male_dir='./spectrogram/Male', female_dir='./spectrogram/Female', img_size=(160, 64), normalize=True):

        print("\nInitializing Dataset...")

        self.normalize = normalize
        self.img_size = img_size

        # Load male and female data
        male_images, male_labels = self.load_images(male_dir, label=1)  # Male images labeled as 1
        female_images, female_labels = self.load_images(female_dir, label=0)  # Female images labeled as 0

        # Split male data into 80% training and 20% testing
        split_idx = int(0.8 * len(male_images))
        self.x_tr, self.y_tr = male_images[:split_idx], male_labels[:split_idx]
        x_te_male, y_te_male = male_images[split_idx:], male_labels[split_idx:]

        # Combine the 20% male data with all female data for testing
        self.x_te = np.concatenate([x_te_male, female_images], axis=0)
        self.y_te = np.concatenate([y_te_male, female_labels], axis=0)

        # Shuffle test data to mix male and female data
        self.x_te, self.y_te = shuffle(self.x_te, self.y_te)

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        print("Number of data\nTraining: %d, Test: %d\n" % (self.num_tr, self.num_te))

        x_sample = self.x_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        try: 
            self.channel = x_sample.shape[2]
        except: 
            self.channel = 1

        self.min_val, self.max_val = x_sample.min(), x_sample.max()

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" % (self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" % (self.min_val, self.max_val))
        print("Normalization: %r" % (self.normalize))
        if self.normalize: 
            print("(from %.3f-%.3f to %.3f-%.3f)" % (self.min_val, self.max_val, 0, 1))

    def load_images(self, directory, label):
        images = []
        labels = []
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(directory, filename)
                image = Image.open(img_path)  
                image = image.resize(self.img_size)
                image = np.array(image)
                images.append(image)
                labels.append(label)  # Assign label (0 for female, 1 for male)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return images, labels

    def reset_idx(self): 
        self.idx_tr, self.idx_te = 0, 0

    def next_train(self, batch_size=1, fix=False):
        start, end = self.idx_tr, self.idx_tr + batch_size
        x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]

        terminator = False
        if end >= self.num_tr:
            terminator = True
            self.idx_tr = 0
            self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else: 
            self.idx_tr = end

        if fix: 
            self.idx_tr = start

        if x_tr.shape[0] != batch_size:
            x_tr, y_tr = self.x_tr[-1 - batch_size:-1], self.y_tr[-1 - batch_size:-1]

        if self.normalize:
            min_x, max_x = x_tr.min(), x_tr.max()
            x_tr = (x_tr - min_x) / (max_x - min_x)

        return x_tr, y_tr, terminator

    def next_test(self, batch_size=1):
        start, end = self.idx_te, self.idx_te + batch_size
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]

        terminator = False
        if end >= self.num_te:
            terminator = True
            self.idx_te = 0
        else: 
            self.idx_te = end

        if self.normalize:
            min_x, max_x = x_te.min(), x_te.max()
            x_te = (x_te - min_x) / (max_x - min_x)

        return x_te, y_te, terminator
