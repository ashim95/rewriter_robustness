import os, sys
import matplotlib.pyplot as plt

colors = ['b', 'r', 'k', 'g', 'c']
markers = ['o', '+', 's', '^', 'x']

if __name__=="__main__":
    plot_data = sys.argv[1]
    column_num = sys.argv[2] # by default take first column as x-axis and this column num as y-axis 'comma separated if multiple lines'
    column_name = sys.argv[3]
    column_numbers = [int(i) for i in column_num.split(',')]
    names = column_name.split(',')
    plot_filename = sys.argv[4]
    x_axis_name = "Ratio"
    y_axis_name = "Accuracy (in %)"
    for i, column_num in enumerate(column_numbers):
        data = []
        x_axis_nums = set()
        with open(plot_data, 'r') as fp:
            for line in fp:
                arr = line.strip().split(',')
                x = float(arr[0])
                y = float(arr[column_num])
                data.append((x, y))

        x_axis_data, y_axis_data = zip(*data)
        plt.plot(x_axis_data, y_axis_data, linestyle='dashed', marker=markers[i], color=colors[i], markersize=12, label=names[i])
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.legend(loc='best')
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
