def plot_histogram(data):
    import matplotlib.pyplot as plt
    plt.hist(data)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()


def plot_scatter(x, y):
    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')
    plt.show()
