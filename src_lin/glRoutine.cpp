#include "glRoutine.h"

GLuint pbo;


void glut_display()
{

}

void glut_idle()
{
    glutPostRedisplay();
}

void glut_reshape( int w, int h )
{
    //rebuild the pixel buffer object
    //re-calculate the dimensions of grids
}

void glut_keyboard( unsigned char key, int x, int y)
{

}

int initPBO()
{
}

int initGL()
{
}