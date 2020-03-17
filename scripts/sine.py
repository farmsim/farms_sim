import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main"""
    times = np.arange(0, 10, 0.001)
    r_1, r_2 = 5, -3
    signal = r_1*(1+np.cos(times)) - r_2*(1+np.cos(times+np.pi))
    plt.plot(times, signal)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
