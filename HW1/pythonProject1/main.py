# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def cal(i, j):
    val = (0.1*i) / (0.1*i + 0.9*j)
    return val


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    val = cal(0.2, 0.001)
    print(val)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
