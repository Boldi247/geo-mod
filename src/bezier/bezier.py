import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from math import factorial

control_points = np.array([
    [[-2.0, -1.0, -1.0], [-1.0, 2.0, 1.0], [1.0, -1.0, -1.0]],
    [[-1.0, -0.5, 0.0], [1.0, 1.0, 1.5], [2.0, -0.5, 0.0]],   
    [[0.0, -1.5, -1.0], [2.0, 2.0, 2.5], [3.0, -1.0, -1.0]]   
])


def bezier_surface(u, v, control_points):
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

def draw_surface(control_points):
    """Draw the Bézier surface."""
    for u in np.linspace(0, 1, 20):
        glBegin(GL_TRIANGLE_STRIP)
        for v in np.linspace(0, 1, 20):
            point = bezier_surface(u, v, control_points)
            glVertex3fv(point)
        glEnd()

def draw_control_points(control_points):
    """Draw control points as blue dots."""
    glColor3f(0.0, 0.0, 1.0)  # Set color to blue
    glPointSize(5)  # Set point size
    glBegin(GL_POINTS)
    for i in range(control_points.shape[0]):
        for j in range(control_points.shape[1]):
            glVertex3fv(control_points[i][j])
    glEnd()

def init_opengl():
    """Initialize OpenGL settings."""
    glClearColor(0.9, 0.9, 0.9, 1.0)  # Set background to light gray
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLight(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
    glLight(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])

def rotate(angle, axis):
    """Rotate the control points by an angle around an axis."""
    global control_points
    angle = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle) + axis[0] ** 2 * (1 - np.cos(angle)),
         axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
         axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
        [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
         np.cos(angle) + axis[1] ** 2 * (1 - np.cos(angle)),
         axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
        [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
         axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
         np.cos(angle) + axis[2] ** 2 * (1 - np.cos(angle))]
    ])
    control_points = np.dot(control_points, rotation_matrix)

def bezier_main():
    """Main function to run the OpenGL window."""
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    init_opengl()

    while True:
        if pygame.key.get_pressed()[K_LEFT]:
            rotate(1, [0, 1, 0])
        if pygame.key.get_pressed()[K_RIGHT]:
            rotate(-1, [0, 1, 0])
        if pygame.key.get_pressed()[K_UP]:
            rotate(1, [1, 0, 0])
        if pygame.key.get_pressed()[K_DOWN]:
            rotate(-1, [1, 0, 0])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_surface(control_points)
        draw_control_points(control_points)  # Draw the control points
        pygame.display.flip()
        pygame.time.wait(10)