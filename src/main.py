import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib import gridspec
from scipy.special import factorial


class CurveEditor3D:
    def __init__(self):
        self.control_points = []
        self.fig = plt.figure(figsize=(16, 8))  # Larger figure size
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])  # Layout for left and right sections
        
        # Left section for adding points, initial points, and display
        self.ax = self.fig.add_subplot(gs[1], projection='3d', position=[0.2, 0.1, 0.65, 0.8])

        # Input TextBoxes for adding points (X, Y, Z)
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

        # Initial control points
        self.initial_points = [
            [[-2.0, -1.0, -1.0], [-1.0, 2.0, 1.0], [1.0, -1.0, -1.0]],
            [[-1.0, -0.5, 0.0], [1.0, 1.0, 1.5], [2.0, -0.5, 0.0]],
            [[0.0, -1.5, -1.0], [2.0, 2.0, 2.5], [3.0, -1.0, -1.0]]
        ]

        self.initial_points = np.array(self.initial_points).reshape(-1, 3).tolist()

        # Initial Points button
        initial_points_ax = plt.axes([0.05, 0.45, 0.2, 0.05])
        self.initial_points_button = Button(initial_points_ax, 'Initial Points')
        self.initial_points_button.on_clicked(self.add_initial_points)

        # Clear button
        clear_ax = plt.axes([0.05, 0.35, 0.2, 0.05])
        self.clear_button = Button(clear_ax, 'Clear Points')
        self.clear_button.on_clicked(self.clear_points)

        # Points Display
        points_display_ax = plt.axes([0.05, 0.05, 0.2, 0.2])
        self.points_display = TextBox(points_display_ax, 'Points', initial='')
        self.update_points_display()

        # Right section for modifying points
        modify_ax = plt.axes([0.85, 0.85, 0.1, 0.075])
        self.modify_button = Button(modify_ax, 'Modify Points')
        self.modify_button.on_clicked(self.modify_or_save_points)
        self.is_modify_mode = True  # Track if we're in modify mode

        # Buttons to switch between surface types
        self.selected_surface_type = 'bezier'

        bezier_btn_ax = plt.axes([0.85, 0.75, 0.1, 0.075])
        self.bezier_button = Button(bezier_btn_ax, 'Bézier')
        self.bezier_button.on_clicked(lambda event: self.set_surface_type('bezier'))

        b_spline_btn_ax = plt.axes([0.85, 0.65, 0.1, 0.075])
        self.b_spline_button = Button(b_spline_btn_ax, 'B-Spline')
        self.b_spline_button.on_clicked(lambda event: self.set_surface_type('b_spline'))

        nurbs_btn_ax = plt.axes([0.85, 0.55, 0.1, 0.075])
        self.nurbs_button = Button(nurbs_btn_ax, 'NURBS')
        self.nurbs_button.on_clicked(lambda event: self.set_surface_type('nurbs'))

        # Create modifiable fields container for X, Y, Z columns
        self.modifiable_fields = []

        # Connect scroll event for zooming
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def set_surface_type(self, event):
        """ Set the selected surface type """
        if event == 'bezier':
            self.selected_surface_type = 'bezier'
        elif event == 'b_spline':
            self.selected_surface_type = 'b_spline'
        elif event == 'nurbs':
            self.selected_surface_type = 'nurbs'
        self.update_plot()
        print("Selected surface type:", self.selected_surface_type)

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

    def modify_or_save_points(self, event):
        """ Toggle between modifying points and saving points """
        if self.is_modify_mode:
            # Switch to save mode
            self.modify_button.label.set_text("Save Points")
            self.show_modifiable_points(event)  # Show input fields for modifying points
            self.is_modify_mode = False
        else:
            # Save the modified points
            self.save_modified_points(event)
            # Switch back to modify mode
            self.modify_button.label.set_text("Modify Points")
            self.is_modify_mode = True

    def show_modifiable_points(self, event):
        """ Show modifiable input fields for existing points in table format """
        self.clear_modifiable_fields()

        # Increase space between rows and increase space between X, Y, Z fields
        for i, point in enumerate(self.control_points):
            x_input_ax = plt.axes([0.75, 0.7 - i * 0.07, 0.05, 0.03])  # Spacing between rows
            x_input = TextBox(x_input_ax, f'X{i+1}', initial=str(point[0]))

            y_input_ax = plt.axes([0.82, 0.7 - i * 0.07, 0.05, 0.03])  # Increased spacing between X and Y
            y_input = TextBox(y_input_ax, f'Y{i+1}', initial=str(point[1]))

            z_input_ax = plt.axes([0.89, 0.7 - i * 0.07, 0.05, 0.03])  # Increased spacing between Y and Z
            z_input = TextBox(z_input_ax, f'Z{i+1}', initial=str(point[2]))

            self.modifiable_fields.append((x_input, y_input, z_input))

        plt.draw()  # Redraw the canvas

    def clear_modifiable_fields(self):
        """ Clear the existing modifiable fields """
        for x_input, y_input, z_input in self.modifiable_fields:
            x_input_ax = x_input.ax
            y_input_ax = y_input.ax
            z_input_ax = z_input.ax
            x_input_ax.remove()  # Remove the X input field
            y_input_ax.remove()  # Remove the Y input field
            z_input_ax.remove()  # Remove the Z input field

        self.modifiable_fields = []
        plt.draw()  # Redraw the canvas to reflect the removal of fields

    def save_modified_points(self, event):
        """ Save the modified points and update the plot """
        for i, (x_input, y_input, z_input) in enumerate(self.modifiable_fields):
            try:
                x = float(x_input.text)
                y = float(y_input.text)
                z = float(z_input.text)
                self.control_points[i] = [x, y, z]
            except ValueError:
                print(f"Invalid input for point {i+1}.")

        self.clear_modifiable_fields()  # Hide the input fields after saving
        self.update_plot()
        self.update_points_display()

    def clear_points(self, event):
        """ Clear all points from the control points list and update the plot """
        self.control_points = []  # Reset control points to an empty list
        self.update_plot()  # Clear the plot
        self.update_points_display()  # Update the display to reflect cleared points

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

        # If there are at least 4 points
        if len(self.control_points) >= 4:
            grid_size = int(np.ceil(np.sqrt(len(self.control_points))))  # Determine grid size (rounded up)
            control_grid = self.form_control_grid(self.control_points, grid_size)

            if self.selected_surface_type == 'bezier':
                self.plot_bezier_surface(control_grid)
            elif self.selected_surface_type == 'b_spline':
                self.plot_b_spline_surface(control_grid)

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
    
    def plot_b_spline_surface(self, control_grid, steps=20):
        """ Plot a B-Spline surface using a grid of control points """
        u_vals = np.linspace(0, 1, steps)
        v_vals = np.linspace(0, 1, steps)
        
        surface_points = np.zeros((steps, steps, 3))

        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                surface_points[i, j] = self.b_spline_surface(u, v, control_grid)

        # Plot the B-Spline surface
        self.ax.plot_surface(surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2], color='cyan', alpha=0.7)

    def b_spline_basis(self, i, k, t, knots):
        """ Recursive Cox-de Boor formula for B-spline basis function """
        if k == 0:
            return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
        else:
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            term1 = ((t - knots[i]) / denom1) * self.b_spline_basis(i, k - 1, t, knots) if denom1 != 0 else 0
            term2 = ((knots[i + k + 1] - t) / denom2) * self.b_spline_basis(i + 1, k - 1, t, knots) if denom2 != 0 else 0

            return term1 + term2

    def b_spline_surface(self, u, v, control_points, degree=3):
        """ Calculate a point on a B-Spline surface """
        n, m = control_points.shape[:2]
        
        # Create knot vectors (uniform knot vector for simplicity)
        knot_vector_u = np.linspace(0, 1, n + degree + 1)
        knot_vector_v = np.linspace(0, 1, m + degree + 1)

        point = np.zeros(3)
        
        # Calculate the surface point using the B-spline basis functions
        for i in range(n):
            for j in range(m):
                basis_u = self.b_spline_basis(i, degree, u, knot_vector_u)
                basis_v = self.b_spline_basis(j, degree, v, knot_vector_v)
                point += basis_u * basis_v * control_points[i][j]
        return point

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
