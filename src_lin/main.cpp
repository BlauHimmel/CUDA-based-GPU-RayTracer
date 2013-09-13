#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "constant.h"

using namespace std;

int main( int argc, char* argv[] )
{
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL );

    glutInitContextVersion( 4, 0 );
    glutInitContextFlags( GLUT_FORWARD_COMPATIBLE );
    glutInitContextProfile( GLUT_CORE_PROFILE );

    glutInitWindowSize( INIT_WIN_WIDTH, INIT_WIN_HEIGHT );
    glutCreateWindow( "GPU RayTracer" );

    GLenum errCode = glewInit();
    if( errCode != GLEW_OK )
    {
        cerr<<"Error: "<<glewGetErrorString(errCode)<<endl;
        return 1;
    }
    return 0;
}

