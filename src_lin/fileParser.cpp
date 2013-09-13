#include "FileParser.h"
#include "Light.h"
#include "shape.h"
#include "sphere.h"
#include "triangle.h"
#include "transform.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <array>
#include "glm/glm.hpp"


using namespace std;
using namespace glm;

// Function to read the input data values
// Use is optional, but should be very helpful in parsing.  
bool readvals(stringstream &s, const int numvals, float* values) 
{
    for (int i = 0; i < numvals; i++) {
        s>>values[i]; 
        if (s.fail()) {
            cout << "Failed reading value " << i << " will skip\n"; 
            return false;
        }
    }
    return true; 
}
void rightmultiply(const mat4 & M, vector<mat4> &transfstack) 
{
    mat4 &T = transfstack.back(); 
    T = T * M; 
}

FileParser::FileParser(void)
{
}


FileParser::~FileParser(void)
{
}

int FileParser::parse( const char input[], SceneDesc& sceneDesc )
{
    fstream inFile;
    string itemName;
    string lineStr;
    string output;
    //stringstream lineSStr;
    char lineBuf[256];
    float param[10];

    vec3 diffuse(0.0f);
    vec3 specular(0.0f);
    vec3 emission(0.0f);
    vec3 ambient(0.0f);
    float shininess = 0;
    
    float attenu_const = 1;
    float attenu_linear = 0;
    float attenu_quadratic = 0;

    vector<mat4> transtack;
    int maxvert = 0;
    int maxnorm = 0;
    vector<vec3> vertices;  //vertices without normal vectors
    vector<vec3> vertnorms; //vertices with normal vectors

    transtack.push_back( mat4(1.0) );
    
    inFile.open( input );
    if( !inFile.is_open() )
        return -1;

    while(1)
    {
        inFile.getline( lineBuf, 255 );
        
        lineStr = lineBuf;
        stringstream lineSStr( lineStr );
        
        if( inFile.eof() )
            break;
        if( lineStr.find_first_not_of( " \t\r\n" ) == string::npos || lineStr[0] == '#' )
            continue;

        lineSStr>>itemName;

        if( itemName == "camera" )
        {
            if( readvals( lineSStr, 10, param ) )
            {
                sceneDesc.eyePos = vec3( param[0], param[1], param[2] );
                sceneDesc.eyePosHomo = vec4( param[0], param[1], param[2], 1.0f );
                sceneDesc.center = vec3( param[3], param[4], param[5] );
                sceneDesc.up = vec3( param[6], param[7], param[8] );
                sceneDesc.fovy = param[9];

                //convert the fovy to radian unit
                sceneDesc.fovy = sceneDesc.fovy * pi /180.0;
            }


        }
        else if( itemName == "size" )
        {
            if( readvals( lineSStr, 2, param ) )
            {
                sceneDesc.width = param[0];
                sceneDesc.height = param[1];
            }
        }
        else if( itemName == "maxdepth" )
        {
            if( readvals( lineSStr, 1, param ) )
            {
                sceneDesc.rayDepth = param[0];
            }
        }
        else if( itemName == "output" )
        {
            lineSStr>>output;
            cout<<output<<endl;
        }
        else if( itemName == "diffuse" )
        {
            if( readvals( lineSStr, 3, param ) )
            {
                diffuse = vec3( param[0], param[1], param[2] );
            }
        }
        else if( itemName == "specular" )
        {
            if( readvals( lineSStr, 3, param ) )
            {
                specular = vec3( param[0], param[1], param[2] );
            }
        }
        else if( itemName == "shininess" )
        {
            if( readvals( lineSStr, 1, param ) )
            {
                shininess = param[0];
            }
        }
        else if( itemName == "emission" )
        {
            if( readvals( lineSStr, 3, param ) )
            {
                emission = vec3( param[0], param[1], param[2] );
            }
        }
        else if( itemName == "directional" || itemName == "point" )
        {
            if( readvals( lineSStr, 6, param ) )
            {
                Light light;
                light.pos = vec4( param[0], param[1], param[2], 1 );
                
                if( itemName != "point" )
                    light.pos[3] = 0;
             
                light.pos = transtack.back() * light.pos ;
                light.color = vec3( param[3], param[4], param[5] );

                light.attenu_const = attenu_const;
                light.attenu_linear = attenu_linear;
                light.attenu_quadratic = attenu_quadratic;

                sceneDesc.lights.push_back( light );
            }
        }
        else if( itemName == "ambient" )
        {
            if( readvals( lineSStr, 3, param ) )
            {
                sceneDesc.ambient = ambient = vec3( param[0], param[1], param[2] ) ;

            }
        }
        else if( itemName == "attenuation" )
        {
            if( readvals( lineSStr, 3, param ) )
            {
                attenu_const = param[0];
                attenu_linear = param[1];
                attenu_quadratic = param[2];

            }
        }
        else if( itemName == "translate" )
        {
            if( readvals( lineSStr, 3, param ) )
            {
                mat4 m = Transform::translate( param[0], param[1], param[2] ) ;
                rightmultiply( m, transtack );
            }
        }
        else if( itemName == "rotate" )
        {
            if( readvals( lineSStr, 4, param ) )
            {
                mat3 m = Transform::rotate( param[3], vec3( param[0], param[1], param[2] ) ) ;
                mat4 m4x4 = mat4( m );
                rightmultiply( m4x4, transtack );
            }
        }
        else if( itemName == "scale" )
        {
            if( readvals( lineSStr, 3, param ) )
            {
                mat4 m = Transform::scale( param[0], param[1], param[2] ) ;
                rightmultiply( m, transtack );
            }
        }
        else if( itemName == "pushTransform" )
        {
            transtack.push_back( transtack.back() );
        }
        else if( itemName == "popTransform" )
        {
            if( transtack.size() <= 1 )
                cout<<"No more transform matrix could be poped\n";
            else
                transtack.pop_back();
        }
        else if( itemName == "sphere" )
        {
            if( readvals( lineSStr, 4, param ) )
            {
                Sphere *pSphere = new Sphere();
                pSphere->center = vec4( param[0], param[1], param[2], 1.0 );
                pSphere->radius = param[3];

                pSphere->transform = transtack.back();
                pSphere->invTrans = inverse( pSphere->transform );

                pSphere->shininess = shininess;
                pSphere->emission = emission;
                pSphere->diffuse = diffuse;
                pSphere->specular = specular;
                pSphere->ambient = ambient;

                sceneDesc.primitives.push_back(pSphere);
             
            }
        }
        else if( itemName == "maxverts" )
        {
            lineSStr>>maxvert;
            vertices.reserve( maxvert );
        }
        else if( itemName == "maxvertnorms" )
        {
            lineSStr>>maxnorm;
            vertnorms.reserve( maxnorm );
        }
        else if( itemName == "vertex" )
        {
            if( maxvert == 0 ) //No vertices, skip the parsing
                continue;
            if( readvals( lineSStr, 3, param ) )
            {
                vec3 vertex( param[0], param[1], param[2] );
                vertices.push_back( vertex );
            }

        }
        else if( itemName == "vertexnormal" )
        {
            if( maxnorm == 0 ) //No vertices, skip the parsing
                continue;
            if( readvals( lineSStr, 6, param ) )
            {
                vec3 vertex( param[0], param[1], param[2] );
                vec3 normal( param[3], param[4], param[5] );
                vertnorms.push_back( vertex );
                vertnorms.push_back( normal );
            }

        }
        else if( itemName == "tri" )
        {
           if( readvals( lineSStr, 3, param ) )
           {
               Triangle *pTri = new Triangle();
               pTri->v[0] = vertices[ (int)param[0] ];
               pTri->v[1] = vertices[ (int)param[1] ];
               pTri->v[2] = vertices[ (int)param[2] ];

               //calculate plane normal
               pTri->pn = normalize( cross( pTri->v[1] - pTri->v[0], pTri->v[2] - pTri->v[0] ) );

               //Homogenuous coordinate representation
               pTri->vv[0] = vec4( vertices[ (int)param[0] ], 1 );
               pTri->vv[1] = vec4( vertices[ (int)param[1] ], 1 );
               pTri->vv[2] = vec4( vertices[ (int)param[2] ], 1 );

               pTri->diffuse = diffuse;
               pTri->emission = emission;
               pTri->specular = specular;
               pTri->shininess = shininess;
               pTri->ambient = ambient;

               pTri->transform = transtack.back();
               pTri->invTrans = inverse( pTri->transform );

               sceneDesc.primitives.push_back( pTri );
           }
        }
        else if( itemName == "trinormal" )
        {
           if( readvals( lineSStr, 6, param ) )
           {
               Triangle *pTri = new Triangle();

               pTri->v[0] = vertices[ (int)param[0] ];
               pTri->v[1] = vertices[ (int)param[1] ];
               pTri->v[2] = vertices[ (int)param[2] ];

               pTri->vv[0] = vec4( vertnorms[ 2*(int)param[0] ], 1 );
               pTri->vv[1] = vec4( vertnorms[ 2*(int)param[1] ], 1 );
               pTri->vv[2] = vec4( vertnorms[ 2*(int)param[2] ], 1 );

               pTri->n[0] = vertnorms[ 2*(int)param[0]+1 ];
               pTri->n[1] = vertnorms[ 2*(int)param[1]+1 ];
               pTri->n[2] = vertnorms[ 2*(int)param[2]+1 ];

               pTri->diffuse = diffuse;
               pTri->emission = emission;
               pTri->specular = specular;
               pTri->shininess = shininess;

               pTri->transform = transtack.back();
               pTri->invTrans = inverse( pTri->transform );

               sceneDesc.primitives.push_back( pTri );
              
           }
        }

        
        
    }

    return 0;
}
