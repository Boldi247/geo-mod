import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib import gridspec
from scipy.special import factorial


class CurveEditor3D:
    def __init__(self):
        self.control_points = []
        self.weights = []
        self.modifiable_fields = []
        self.headers = []
        self.fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

        # Left section for adding points, initial points, and display
        self.ax = self.fig.add_subplot(gs[1], projection='3d', position=[0.2, 0.1, 0.65, 0.8])

        # Set axis limits before anything is plotted
        self.set_axis_limits()

        # Input TextBoxes for adding points (X, Y, Z)
        x_input_ax = plt.axes([0.05, 0.75, 0.2, 0.05])
        self.x_input = TextBox(x_input_ax, 'X Coord')

        y_input_ax = plt.axes([0.05, 0.65, 0.2, 0.05])
        self.y_input = TextBox(y_input_ax, 'Y Coord')

        z_input_ax = plt.axes([0.05, 0.55, 0.2, 0.05])
        self.z_input = TextBox(z_input_ax, 'Z Coord')

        # Weight input for NURBS
        weight_input_ax = plt.axes([0.05, 0.45, 0.2, 0.05])
        self.weight_input = TextBox(weight_input_ax, 'Weight', initial='1.0')
        self.weight_input.set_active(False) 

        # Add Point button
        add_ax = plt.axes([0.05, 0.85, 0.2, 0.075])
        self.add_button = Button(add_ax, 'Add Point')
        self.add_button.on_clicked(self.add_point)

        # Initial control points for Bézier and B-Spline
        self.initial_points = [
            [[-5., -5., 0.], [0., -5., 5.], [5., -5., 0.]],
            [[-5., 0., 5.], [0., 0., 10.], [5., 0., 5.]],
            [[-5., 5., 0.], [0., 5., 5.], [5., 5., 0.]]
        ]
        self.initial_points = np.array(self.initial_points).reshape(-1, 3).tolist()

        # Initial control points for NURBS with different weights
        self.nurbs_initial_points = self.initial_points
        self.nurbs_weights = [3.0, 2.0, 5.0, 2.0, 3.0, 2.0, 4.0, 2.0, 6.0]

        # Initial Points button for uniform weights
        initial_points_ax = plt.axes([0.05, 0.35, 0.2, 0.05])
        self.initial_points_button = Button(initial_points_ax, 'Uniform Weights')
        self.initial_points_button.on_clicked(self.add_uniform_initial_points)

        # Initial Points button for NURBS-specific weights
        nurbs_points_ax = plt.axes([0.05, 0.25, 0.2, 0.05])
        self.nurbs_points_button = Button(nurbs_points_ax, 'Different Weights')
        self.nurbs_points_button.on_clicked(self.add_nurbs_initial_points)

        # Clear button
        clear_ax = plt.axes([0.05, 0.15, 0.2, 0.05])
        self.clear_button = Button(clear_ax, 'Clear Points')
        self.clear_button.on_clicked(self.clear_points)

        # Points Display
        points_display_ax = plt.axes([0.05, 0.05, 0.2, 0.075])
        self.points_display = TextBox(points_display_ax, 'Points', initial='')

        # Right section for modifying points
        modify_ax = plt.axes([0.85, 0.85, 0.1, 0.075])
        self.modify_button = Button(modify_ax, 'Modify Points')
        self.modify_button.on_clicked(self.modify_or_save_points)
        self.is_modify_mode = True

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

        self.update_points_display()

    def set_surface_type(self, event):
        """ Set the selected surface type and handle weight input visibility """
        self.selected_surface_type = event
        print("Selected surface type:", self.selected_surface_type)

        # Show or hide weight input depending on surface type
        self.weight_input.set_active(self.selected_surface_type == 'nurbs')
        self.update_plot()

    def add_uniform_initial_points(self, event):
        """ Add initial points with uniform weights """
        self.control_points = self.initial_points
        self.weights = [1.0] * len(self.control_points)
        self.update_plot()
        self.update_points_display()
    
    def add_nurbs_initial_points(self, event):
        """ Add initial points with different weights """
        self.control_points = self.nurbs_initial_points
        self.weights = self.nurbs_weights
        self.update_plot()
        self.update_points_display()

    def add_point(self, event):
        """ Add a point based on input from text boxes """
        try:
            x = float(self.x_input.text)
            y = float(self.y_input.text)
            z = float(self.z_input.text)
            weight = float(self.weight_input.text) if self.selected_surface_type == 'nurbs' else 1.0

            new_point = [x, y, z]
            self.control_points.append(new_point) 
            self.weights.append(weight)
            self.update_plot()
            self.update_points_display()

            # Clear the input boxes after the point is added
            self.x_input.set_val('')
            self.y_input.set_val('')
            self.z_input.set_val('')
            self.weight_input.set_val('1.0')
        except ValueError:
            print("Please enter valid numerical values for the coordinates.")

    def modify_or_save_points(self, event):
        """ Toggle between modifying points and saving points """
        if self.is_modify_mode:
            # Switch to save mode
            self.modify_button.label.set_text("Save Points")
            self.show_modifiable_points() 
            self.is_modify_mode = False
        else:
            # Save the modified points
            self.save_modified_points()
            # Switch back to modify mode
            self.modify_button.label.set_text("Modify Points")
            self.is_modify_mode = True

    def show_modifiable_points(self):
        """ Show modifiable input fields for existing points in table format """
        self.clear_modifiable_fields()
        self.modifiable_fields = []
        self.headers = []

        # Add headers for the table
        header_ax = plt.axes([0.72, 0.5, 0.25, 0.03])
        header_ax.axis("off")
        header_ax.text(0.1, 0.5, "X", fontsize=10, ha='center', transform=header_ax.transAxes)
        header_ax.text(0.35, 0.5, "Y", fontsize=10, ha='center', transform=header_ax.transAxes)
        header_ax.text(0.6, 0.5, "Z", fontsize=10, ha='center', transform=header_ax.transAxes)
        header_ax.text(0.85, 0.5, "W", fontsize=10, ha='center', transform=header_ax.transAxes)
        self.headers.append(header_ax)

        # Create input fields for each control point and weight
        for i, (point, weight) in enumerate(zip(self.control_points, self.weights)):
            y_pos = 0.45 - i * 0.07

            x_input_ax = plt.axes([0.72, y_pos, 0.05, 0.03])
            x_input = TextBox(x_input_ax, '', initial=str(point[0]))

            y_input_ax = plt.axes([0.79, y_pos, 0.05, 0.03])
            y_input = TextBox(y_input_ax, '', initial=str(point[1]))

            z_input_ax = plt.axes([0.85, y_pos, 0.05, 0.03])
            z_input = TextBox(z_input_ax, '', initial=str(point[2]))

            weight_input_ax = plt.axes([0.91, y_pos, 0.05, 0.03])
            weight_input = TextBox(weight_input_ax, '', initial=str(weight))

            self.modifiable_fields.append((x_input, y_input, z_input, weight_input))

        plt.draw()

    def clear_modifiable_fields(self):
        """ Clear the existing modifiable fields and headers """
        for inputs in self.modifiable_fields:
            for input_field in inputs:
                input_field.ax.remove()

        # Remove headers
        for header in self.headers:
            header.remove()

        self.modifiable_fields = []
        self.headers = []
        plt.draw()  # Redraw the canvas to reflect the removal of fields

    def save_modified_points(self):
        """ Save the modified points and update the plot """
        new_points = []
        new_weights = []

        for inputs in self.modifiable_fields:
            try:
                x = float(inputs[0].text)
                y = float(inputs[1].text)
                z = float(inputs[2].text)
                weight = float(inputs[3].text)
                new_points.append([x, y, z])
                new_weights.append(weight)
            except ValueError:
                print("Invalid input for a control point or weight.")

        self.control_points = new_points
        self.weights = new_weights

        self.clear_modifiable_fields()
        self.update_plot()
        self.update_points_display()

    def clear_points(self, event):
        """ Clear all points from the control points list and update the plot """
        self.control_points = []
        self.update_plot() 
        self.update_points_display() 

    def set_axis_limits(self):
        """ Set constant axis limits from -10 to 10 in all directions """
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])


    def update_plot(self):
        """ Update the plot with new control points """
        self.ax.cla() 
        self.set_axis_limits()

        if len(self.control_points) < 1:
            return

        # Draw lines connecting control points
        if len(self.control_points) >= 2:
            for i, point1 in enumerate(self.control_points):
                for j, point2 in enumerate(self.control_points):
                    if i != j:
                        x_vals = [point1[0], point2[0]]
                        y_vals = [point1[1], point2[1]]
                        z_vals = [point1[2], point2[2]]
                        self.ax.plot(x_vals, y_vals, z_vals, color='yellow', linewidth=1)

        # Plot the selected surface type if enough points are available
        if len(self.control_points) >= 4:
            grid_size = int(np.ceil(np.sqrt(len(self.control_points))))
            control_grid = self.form_control_grid(self.control_points, grid_size)

            if self.selected_surface_type == 'bezier':
                self.plot_bezier_surface(control_grid)
            elif self.selected_surface_type == 'b_spline':
                self.plot_b_spline_surface(control_grid)
            elif self.selected_surface_type == 'nurbs':
                self.plot_nurbs_surface(control_grid)

        plt.draw()


    def update_points_display(self):
        """ Update the display of the control points """
        points_str = "\n".join([f"({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}, w={w:.2f})"
                                for x, w in zip(self.control_points, self.weights)])
        self.points_display.set_val(points_str)

    def on_scroll(self, event):
        """ Handle the mouse scroll event for zooming """
        # Zoom in for scroll up, out for scroll down
        scale_factor = 0.9 if event.button == 'up' else 1.1 

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

        plt.draw()

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
        """Plot a B-Spline surface using a grid of control points and show individual points."""
        u_vals = np.linspace(0, 1, steps)
        v_vals = np.linspace(0, 1, steps)

        surface_points = []

        # Compute B-Spline surface points
        for u in u_vals:
            row = []
            for v in v_vals:
                point = self.b_spline_surface(u, v, control_grid)
                if point is not None:
                    row.append(point)
                else:
                    row.append([np.nan, np.nan, np.nan]) 
            surface_points.append(row)

        surface_points = np.array(surface_points)

        # Extract X, Y, Z coordinates
        X = surface_points[:, :, 0]
        Y = surface_points[:, :, 1]
        Z = surface_points[:, :, 2]

        # Plot the B-Spline surface
        self.ax.plot_surface(X, Y, Z, color='cyan', alpha=0.7, edgecolor='none')

        # Plot individual points on the surface
        valid_points = ~np.isnan(X)
        self.ax.scatter(
            X[valid_points], Y[valid_points], Z[valid_points],
            color='red', s=10, label='Surface Points'
        )

        # Highlight control points
        control_points_flat = [p for row in control_grid for p in row]
        self.ax.scatter(
            *zip(*control_points_flat),
            color='black', s=20, label='Control Points'
        )

        # Add a legend to differentiate points
        self.ax.legend()


    def b_spline_surface(self, u, v, control_points, degree=3):
        """Calculate a point on a B-Spline surface."""
        n, m = control_points.shape[:2]

        knot_vector_u = np.linspace(0, 1, n + degree + 1)
        knot_vector_v = np.linspace(0, 1, m + degree + 1)

        point = np.zeros(3)

        # Calculate the surface point using the B-spline basis functions (no weights)
        weight_sum = 0.0 

        for i in range(n):
            for j in range(m):
                basis_u = self.b_spline_basis(i, degree, u, knot_vector_u)
                basis_v = self.b_spline_basis(j, degree, v, knot_vector_v)
                weight = basis_u * basis_v
                point += weight * control_points[i][j]
                weight_sum += weight

        # If the weight sum is too small, return None (to avoid zero points)
        if weight_sum < 1e-6:
            print(f"Invalid surface point at (u, v) = ({u}, {v}), weight sum = {weight_sum}")
            return None

        # Normalize the point by weight sum
        return point / weight_sum


    def b_spline_basis(self, i, k, t, knots):
        """ Recursive Cox-de Boor formula for B-spline basis function """
        if k == 0:
            return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
        else:
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            # Avoid division by zero or very small values
            if denom1 == 0 or denom2 == 0:
                return 0.0

            term1 = ((t - knots[i]) / denom1) * self.b_spline_basis(i, k - 1, t, knots)
            term2 = ((knots[i + k + 1] - t) / denom2) * self.b_spline_basis(i + 1, k - 1, t, knots)

            return term1 + term2

    def plot_bezier_surface(self, control_grid, steps=20):
        """Plot a Bézier surface using a grid of control points and show individual points."""
        u = np.linspace(0, 1, steps)
        v = np.linspace(0, 1, steps)
        U, V = np.meshgrid(u, v)
        surface_points = np.zeros((steps, steps, 3))

        # Compute Bézier surface points
        for i in range(steps):
            for j in range(steps):
                u_val = U[i, j]
                v_val = V[i, j]
                surface_points[i, j] = self.bezier_surface(u_val, v_val, control_grid)

        # Extract X, Y, Z coordinates for plotting
        X = surface_points[:, :, 0]
        Y = surface_points[:, :, 1]
        Z = surface_points[:, :, 2]

        # Plot the Bézier surface
        self.ax.plot_surface(X, Y, Z, color='blue', alpha=0.6, edgecolor='none')

        # Plot individual points on the surface
        self.ax.scatter(
            X.flatten(), Y.flatten(), Z.flatten(),
            color='red', s=10, label='Surface Points'
        )

        # Optionally, highlight control points as well
        control_points_flat = [p for row in control_grid for p in row]
        self.ax.scatter(
            *zip(*control_points_flat),
            color='black', s=20, label='Control Points'
        )

        # Add a legend to differentiate points
        self.ax.legend()


    def bezier_surface(self, u, v, control_grid):
        """ Compute a point on a Bézier surface """
        n, m = control_grid.shape[:2]
        point = np.zeros(3)

        for i in range(n):
            for j in range(m):
                bernstein_u = self.bernstein(i, n - 1, u)
                bernstein_v = self.bernstein(j, m - 1, v)
                point += bernstein_u * bernstein_v * control_grid[i, j]

        return point

    def bernstein(self, i, n, t):
        """ Compute the Bernstein polynomial value """
        return factorial(n) / (factorial(i) * factorial(n - i)) * (t ** i) * ((1 - t) ** (n - i))

    def plot_nurbs_surface(self, control_grid, steps=20):
        """Plot a NURBS surface using a grid of control points and show individual points."""
        u_vals = np.linspace(0, 1, steps)
        v_vals = np.linspace(0, 1, steps)

        surface_points = np.full((steps, steps, 3), np.nan)

        # Compute NURBS surface points
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                point = self.nurbs_surface(u, v, control_grid, self.weights)
                if point is not None:
                    surface_points[i, j] = point

        # Extract X, Y, Z coordinates, ignoring NaNs
        X = surface_points[:, :, 0]
        Y = surface_points[:, :, 1]
        Z = surface_points[:, :, 2]

        # Plot the NURBS surface
        self.ax.plot_surface(X, Y, Z, color='magenta', alpha=0.7, edgecolor='none')

        # Plot individual points on the surface
        valid_points = ~np.isnan(X)
        self.ax.scatter(
            X[valid_points], Y[valid_points], Z[valid_points],
            color='red', s=10, label='Surface Points'
        )

        # Highlight control points
        control_points_flat = [p for row in control_grid for p in row]
        self.ax.scatter(
            *zip(*control_points_flat),
            color='black', s=20, label='Control Points'
        )

        # Add a legend to differentiate points
        self.ax.legend()


    def nurbs_surface(self, u, v, control_points, weights, degree=3):
        """Calculate a point on a NURBS surface."""
        n, m = control_points.shape[:2]

        knot_vector_u = np.linspace(0, 1, n + degree + 1)
        knot_vector_v = np.linspace(0, 1, m + degree + 1)

        point = np.zeros(3)
        weight_sum = 0.0

        # Calculate the surface point using the B-spline basis functions and weights
        for i in range(n):
            for j in range(m):
                basis_u = self.b_spline_basis(i, degree, u, knot_vector_u)
                basis_v = self.b_spline_basis(j, degree, v, knot_vector_v)
                weight = weights[i * m + j]
                point += basis_u * basis_v * weight * control_points[i][j]
                weight_sum += basis_u * basis_v * weight

        # Avoid adding invalid points
        return point / weight_sum if weight_sum > 1e-6 else None


# Main application
def main():
    editor = CurveEditor3D()
    plt.show()


if __name__ == '__main__':
    main()
