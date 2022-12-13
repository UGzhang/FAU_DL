
from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator


if __name__ == '__main__':

    checker = Checker(16, 2)
    checker.draw()
    checker.show()

    circle = Circle(256, 25, (32,32))
    circle.draw()
    circle.show()

    sep = Spectrum(256)
    sep.draw()
    sep.show()

    img = ImageGenerator("exercise_data", "Labels.json", 12, (32,32,3), True, True, True)
    img.show()

