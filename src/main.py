import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox


class CurveEditor3D:
    def __init__(self):
        self.control_points = []
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d', position=[0.3, 0.1, 0.65, 0.8])
        self.adding_point = False  # Track if we're in 'add point' mode
        self.axis_selection = 0  # 0 = x, 1 = y, 2 = z
        self.new_point = [0, 0, 0]  # To store the new point coordinates

        # Add Point button
        add_ax = plt.axes([0.05, 0.85, 0.2, 0.075])  # Position for the Add Point button
        self.add_button = Button(add_ax, 'Add Point')
        self.add_button.on_clicked(self.initiate_add_point)

        # Points Display
        points_display_ax = plt.axes([0.05, 0.05, 0.2, 0.2])
        self.points_display = TextBox(points_display_ax, 'Points', initial='')

        # Connect the event handler for clicking on the plot and scrolling
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

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
                self.new_point[2] = event.ydata
                print(f"Z selected: {self.new_point[2]}")
                self.control_points.append(self.new_point.copy())
                self.new_point = [0, 0, 0]  # Reset for the next point
                self.axis_selection = 0  # Start from x-axis again
                self.adding_point = False  # Disable adding point mode
                self.update_plot()  # Update the plot after adding the new point
                self.update_points_display()  # Update points display

    def update_plot(self):
        """ Update the plot with new control points """
        if len(self.control_points) < 1:
            return

        self.ax.cla()  # Clear the previous plot
        self.ax.scatter(*zip(*self.control_points), color='black')  # Draw control points

        plt.draw()

    def update_points_display(self):
        """ Update the display of the control points """
        points_str = "\n".join([f"({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})" for x in self.control_points])
        # Update the display box
        self.points_display.set_val(points_str)

    def on_scroll(self, event):
        """ Handle mouse scroll for zooming in/out on the plot """
        scale_factor = 0.9 if event.button == 'up' else 1.1  # Zoom in on scroll up, out on scroll down

        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()

        # Calculate the new limits by scaling the current limits
        new_xlim = self.scale_axis(xlim, scale_factor)
        new_ylim = self.scale_axis(ylim, scale_factor)
        new_zlim = self.scale_axis(zlim, scale_factor)

        # Apply the new axis limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.ax.set_zlim(new_zlim)

        plt.draw()  # Redraw the plot with new scaling

    def scale_axis(self, axis_limits, scale_factor):
        """ Scale axis limits by a scale factor """
        midpoint = (axis_limits[0] + axis_limits[1]) / 2.0
        range_ = (axis_limits[1] - axis_limits[0]) * scale_factor / 2.0
        return [midpoint - range_, midpoint + range_]


# Main application
def main():
    editor = CurveEditor3D()
    plt.show()


if __name__ == '__main__':
    main()
