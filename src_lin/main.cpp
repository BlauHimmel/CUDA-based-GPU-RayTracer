#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "constant.h"
#include "cudaRaytracer.h"
#include "SceneDesc.h"
#include "FileParser.h"
#include "ColorImage.h"
#include "glRoutine.h"

using namespace std;

const char* IOFILES[6][2] = {
    "scene4-ambient.test", "scene4-ambient.ppm",
    "scene4-specular.test", "scene4-specular.ppm",
    "scene4-emission.test", "scene4-emission.ppm",
    "scene4-diffuse.test", "scene4-diffuse.ppm",
    "scene-diffuse-boxlight.test", "scene-diffuse-boxlight.ppm" ,
    "scene7.test","scene7.test" 
};

int main( int argc, char* argv[] )
{
    //glutInit( &argc, argv );
    //glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL );

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

    glutDisplayFunc( glut_display );
    glutReshapeFunc( glut_reshape );
    glutKeyboardFunc( glut_keyboard );
    glutIdleFunc( glut_idle );


    
    RayTracer* cudaRayTracer = new CudaRayTracer();
    SceneDesc theScene(INIT_WIN_WIDTH,INIT_WIN_HEIGHT);
    ColorImage outputImage;

    FileParser::parse( IOFILES[4][0], theScene );
    outputImage.init( theScene.width, theScene.height );
    
    cudaRayTracer->renderImage( theScene, outputImage );

    outputImage.outputPPM( IOFILES[4][1] );

    return 0;
}

