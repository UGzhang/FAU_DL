import numpy as np
import matplotlib.pyplot as plt


'''
A tile block is consist of:
    black white
    white black

'''
class Checker:
    def __init__(self, res, tileSize):
        self.res = res
        self.tileSize = tileSize
        self.output = np.empty((self.res, self.res), dtype=bool)

    def draw(self):
        tileBlockSize = int(self.res / self.tileSize / 2)

        black = np.zeros([self.tileSize, self.tileSize], dtype=np.bool)
        white = np.ones([self.tileSize, self.tileSize],dtype=np.bool)

        # [black white]
        blackAndWhite = np.concatenate((black, white), axis=1)
        # [white black]
        whiteAndBlack = np.concatenate((white, black), axis=1)

        # [black white]
        # [white block]
        tileBlock = np.concatenate((blackAndWhite, whiteAndBlack), axis=0)

        checker = np.tile(tileBlock, (tileBlockSize, tileBlockSize))
        self.output = checker.copy()

        return checker.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Circle:
    def __init__(self, res, radius, pos):
        self.res = res
        self.radius = radius
        self.pos = pos
        self.output = np.empty((self.res, self.res), dtype=bool)

    def draw(self):
        # a row
        x = np.arange(self.res).reshape(1, self.res)
        # a col
        y = np.arange(self.res).reshape(self.res, 1)

        dis = np.sqrt(np.power(x - self.pos[0], 2) + np.power(y - self.pos[1], 2))
        color = dis <= self.radius
        self.output = color
        return color.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()


'''
    R: 0 -> 1 (row), concatenate n row
    
    G: 0 -> 1 (cols), concatenate n cols
    
    B: 1 -> 0 (row), concatenate n row

'''
class Spectrum:
    def __init__(self, res):
        self.res = res
        self.output = np.empty((self.res, self.res, 3), dtype=np.float)

    def draw(self):
        color = np.empty((self.res, self.res, 3), dtype=np.float)
        # 1 row -> self.res rows
        color[:, :, 0] = np.tile(np.linspace(0, 1, num=self.res), (self.res, 1))  # R
        # 1 col -> self.res cols
        color[:, :, 1] = np.tile(np.linspace(0, 1, num=self.res).reshape(-1, 1), (1, self.res))  # G
        # 1 col -> self.res cols
        color[:, :, 2] = np.tile(np.linspace(1, 0, num=self.res), (self.res, 1))  # B
        self.output = color
        return color.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()








