import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class CurveEditor3D:
    def __init__(self):
        self.control_points = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.adding_point = False  # Track if we're in 'add point' mode
        self.axis_selection = 0  # 0 = x, 1 = y, 2 = z
        self.new_point = [0, 0, 0]  # To store the new point coordinates

        # Add a button to trigger point addition
        add_ax = plt.axes([0.7, 0.05, 0.1, 0.075])  # Position of the button
        self.add_button = Button(add_ax, 'Add Point')
        self.add_button.on_clicked(self.initiate_add_point)

        # Connect the event handler for clicking on the plot
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def initiate_add_point(self, event):
        """ Enable adding a new point on the plot """
        print("Click on the plot to add a new point.")
        self.adding_point = True  # Enable the point adding process

    def on_click(self, event):
        """ Handle mouse clicks for adding points """
        if self.adding_point and event.inaxes == self.ax:
            # For each click, we're going to store x, y, and z interactively
            if self.axis_selection == 0:  # x-axis
                self.new_point[0] = event.xdata
                print(f"X selected: {self.new_point[0]}")
                self.axis_selection = 1
            elif self.axis_selection == 1:  # y-axis
                self.new_point[1] = event.ydata
                print(f"Y selected: {self.new_point[1]}")
                self.axis_selection = 2
            elif self.axis_selection == 2:  # z-axis (use ydata as proxy)
                # Using ydata as a proxy for z-axis since matplotlib doesn't give us direct z coordinates from clicks.
                self.new_point[2] = event.ydata
                print(f"Z selected: {self.new_point[2]}")
                self.control_points.append(self.new_point.copy())
                self.new_point = [0, 0, 0]  # Reset for the next point
                self.axis_selection = 0  # Start from x-axis again
                self.adding_point = False  # Disable adding point mode
                self.update_plot()  # Update the plot after adding the new point

    def update_plot(self):
        """ Update the plot with new control points """
        if len(self.control_points) < 1:
            return

        self.ax.cla()  # Clear the previous plot
        self.ax.scatter(*zip(*self.control_points), color='black')  # Draw control points

        plt.draw()


# Main application
def main():
    editor = CurveEditor3D()
    plt.show()


if __name__ == '__main__':
    main()
