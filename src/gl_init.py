import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

def initGlut():
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
    glut.glutCreateWindow("Hello, world!")
    glut.glutDisplayFunc(display)
    glut.glutReshapeFunc(reshape)
    glut.glutMainLoop()

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glColor3f(1.0, 1.0, 1.0)
    gl.glBegin(gl.GL_POLYGON)
    gl.glVertex2f(-0.5, -0.5)
    gl.glVertex2f(-0.5, 0.5)
    gl.glVertex2f(0.5, 0.5)
    gl.glVertex2f(0.5, -0.5)
    gl.glEnd()
    gl.glFlush()
    glut.glutSwapBuffers()

def reshape(w, h):
    gl.glViewport(0, 0, w, h)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)