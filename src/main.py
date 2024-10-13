import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from scipy.special import factorial


class CurveEditor3D:
    def __init__(self):
        self.control_points = []
        self.fig = plt.figure(figsize=(12, 6))
        self.ax = self.fig.add_subplot(111, projection='3d', position=[0.3, 0.1, 0.65, 0.8])

        # Input TextBoxes for X, Y, and Z coordinates
        x_input_ax = plt.axes([0.05, 0.75, 0.2, 0.05])
        self.x_input = TextBox(x_input_ax, 'X Coord')

        y_input_ax = plt.axes([0.05, 0.65, 0.2, 0.05])
        self.y_input = TextBox(y_input_ax, 'Y Coord')

        z_input_ax = plt.axes([0.05, 0.55, 0.2, 0.05])
        self.z_input = TextBox(z_input_ax, 'Z Coord')

        # Add Point button
        add_ax = plt.axes([0.05, 0.85, 0.2, 0.075])
        self.add_button = Button(add_ax, 'Add Point')
        self.add_button.on_clicked(self.add_point)

        # Initial controlpoints button
        self.initial_points = ([
            [[-2.0, -1.0, -1.0], [-1.0, 2.0, 1.0], [1.0, -1.0, -1.0]],
            [[-1.0, -0.5, 0.0], [1.0, 1.0, 1.5], [2.0, -0.5, 0.0]],   
            [[0.0, -1.5, -1.0], [2.0, 2.0, 2.5], [3.0, -1.0, -1.0]]   
        ])

        #transform this array to a 1D array
        self.initial_points = np.array(self.initial_points).reshape(-1, 3).tolist()

        #create button
        initial_points_ax = plt.axes([0.05, 0.45, 0.2, 0.05])
        self.initial_points_button = Button(initial_points_ax, 'Initial Points')
        self.initial_points_button.on_clicked(self.add_initial_points)

        # Points Display
        points_display_ax = plt.axes([0.05, 0.05, 0.2, 0.2])
        self.points_display = TextBox(points_display_ax, 'Points', initial='')
        self.update_points_display()

        # Connect scroll event for zooming
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def add_initial_points(self, event):
        """ Add initial points to the control points list """
        self.control_points = self.initial_points
        self.update_plot()
        self.update_points_display()

    def add_point(self, event):
        """ Add a point based on input from text boxes """
        try:
            x = float(self.x_input.text)
            y = float(self.y_input.text)
            z = float(self.z_input.text)

            new_point = [x, y, z]
            self.control_points.append(new_point)  # Add the new point
            self.update_plot()  # Update the plot with the new point
            self.update_points_display()  # Update the list of points

            # Clear the input boxes after the point is added
            self.x_input.set_val('')
            self.y_input.set_val('')
            self.z_input.set_val('')
        except ValueError:
            print("Please enter valid numerical values for the coordinates.")

    def update_plot(self):
        """ Update the plot with new control points and Bézier surface if there are enough points """
        self.ax.cla()  # Clear the previous plot

        if len(self.control_points) < 1:
            return

        # Draw the control points
        self.ax.scatter(*zip(*self.control_points), color='black', label='Control Points')

        # Draw lines connecting control points
        if len(self.control_points) >= 2:
            for i, point1 in enumerate(self.control_points):
                for j, point2 in enumerate(self.control_points):
                    if i != j:
                        x_vals = [point1[0], point2[0]]
                        y_vals = [point1[1], point2[1]]
                        z_vals = [point1[2], point2[2]]
                        self.ax.plot(x_vals, y_vals, z_vals, color='yellow', linewidth=1)

        # If there are at least 4 points, plot the Bézier surface
        if len(self.control_points) >= 4:
            grid_size = int(np.ceil(np.sqrt(len(self.control_points))))  # Determine grid size (rounded up)
            control_grid = self.form_control_grid(self.control_points, grid_size)
            self.plot_bezier_surface(control_grid)

        plt.draw()

    def update_points_display(self):
        """ Update the display of the control points """
        points_str = "\n".join([f"({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})" for x in self.control_points])
        self.points_display.set_val(points_str)

    def on_scroll(self, event):
        """ Handle the mouse scroll event for zooming """
        scale_factor = 0.9 if event.button == 'up' else 1.1  # Zoom in for scroll up, out for scroll down

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

    def form_control_grid(self, control_points, grid_size):
        """ Form a square control point grid from a list of control points """
        num_points = len(control_points)
        filled_points = control_points.copy()

        # If the number of points isn't enough to fill the grid, duplicate the last point
        while len(filled_points) < grid_size * grid_size:
            filled_points.append(control_points[-1])

        # Reshape the control points to form a grid
        return np.array(filled_points).reshape(grid_size, grid_size, 3)

    def plot_bezier_surface(self, control_grid, steps=20):
        """ Plot a Bézier surface using a grid of control points """
        u = np.linspace(0, 1, steps)
        v = np.linspace(0, 1, steps)
        U, V = np.meshgrid(u, v)
        surface_points = np.zeros((steps, steps, 3))

        for i in range(steps):
            for j in range(steps):
                u_val = U[i, j]
                v_val = V[i, j]
                surface_points[i, j] = self.bezier_surface(u_val, v_val, control_grid)

        # Plot the Bézier surface
        self.ax.plot_surface(surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2], color='cyan', alpha=0.7)

    def bezier_surface(self, u, v, control_points):
        """Calculate a point on a Bézier surface."""
        n = control_points.shape[0] - 1
        m = control_points.shape[1] - 1

        point = np.zeros(3)
        for i in range(n + 1):
            for j in range(m + 1):
                # Bernstein polynomial
                bernstein = (factorial(n) / (factorial(i) * factorial(n - i))) * \
                            (factorial(m) / (factorial(j) * factorial(m - j))) * \
                            (u ** i) * ((1 - u) ** (n - i)) * \
                            (v ** j) * ((1 - v) ** (m - j))
                point += bernstein * control_points[i][j]

        return point


# Main application
def main():
    editor = CurveEditor3D()
    plt.show()


if __name__ == '__main__':
    main()
