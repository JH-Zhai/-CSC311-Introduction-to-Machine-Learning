# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def one_a():
    averageSquaredEuclideanDistance = []
    standardDeviationOfTheSquaredEuclideanDistances = []
    for i in range(11):
        d = pow(2, i)
        points = []
        for j in range(100):
            point = numpy.random.rand(d)
            points.append(point)
        squaredEuclideandistances = []
        for m in range(len(points)-1):
            for n in range(m + 1, len(points)):
                dis = 0
                for l in range(d):
                    dis += pow((points[m][l] - points[n][l]), 2)
                squaredEuclideandistances.append(dis)
        average = numpy.mean(squaredEuclideandistances)
        std = numpy.std(squaredEuclideandistances)
        averageSquaredEuclideanDistance.append(average)
        standardDeviationOfTheSquaredEuclideanDistances.append(std)
    print(averageSquaredEuclideanDistance)
    print(standardDeviationOfTheSquaredEuclideanDistances)
    x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    plt.plot(x,averageSquaredEuclideanDistance)
    plt.xlabel('Dimension')
    plt.ylabel('averageSquaredEuclideanDistance')
    plt.show()
    plt.plot(x,standardDeviationOfTheSquaredEuclideanDistances)
    plt.xlabel('Dimension')
    plt.ylabel('standardDeviationOfTheSquaredEuclideanDistances')
    plt.show()









        # print(d)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # one_a()
    # a = [0,1,2,3]
    # print(len(a))
    # print_hi('PyCharm')
    # for i in range(1,4+1):
    #     print(i)
    # i = numpy.random.rand(5)
    # print(i)
    # x = [0, 2,5,9]
    # y = [56,22,56,90]
    # plt.plot(x,y)
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
