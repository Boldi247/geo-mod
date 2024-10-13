# TODO :: Correct the point draw (A pontok lerakása nem az igazi
#  legalábbis funckiójában működik csak a 2D ről 3D re való konvertálás
#  nem tetszik nekem - Ádám)

import numpy as np
import matplotlib.pyplot as plt
import curve_editor as ce



# Main application
def main():
    ce.CurveEditor3D()
    plt.show()


if __name__ == '__main__':
    main()
