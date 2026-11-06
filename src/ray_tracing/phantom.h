////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// sets voxelized phantoms of 3D geometric shapes to assist in algorithm
// development and testing
////////////////////////////////////////////////////////////////////////////////
#ifndef __PHANTOM_H
#define __PHANTOM_H

#ifdef WIN32
#pragma once
#endif

#include <string>
#include <stdlib.h>
#include "parameters.h"
using namespace std;

/**
 * This class provides CPU-based implementations (accelerated by OpenMP) to specify voxelized phantoms as a collection of geometric shapes.
 */

class geometricObject
{
public:
    geometricObject();
    geometricObject(int type_in, float* c_in, float* r_in, float val_in, float* A_in, float* clip_in, const char* chemForm_in);
    ~geometricObject();

    /**
     * \fn          reset
     * \brief       deletes all data and restores default values
     */
    void reset();

    /**
     * \fn          init
     * \brief       sets all member variables
     * \param[in]   type: the object type
     * \param[in]   c: center of the object
     * \param[in]   r: radii of the object
     * \param[in]   val: the density (or arbitrary value) to proscribe inside the object
     * \param[in]   A: 3D rotation matrix
     * \param[in]   clip: pointer to array that describes the clipping plane(s)
     * \param[in]   chemForm: (optional) chemical formula of this material
     * \return      true if successful, false otherwise
     */
    bool init(int type, float* c, float* r, float val, float* A = NULL, float* clip = NULL, const char* chemForm = NULL);

    /**
     * \fn          intersectionEndPoints
     * \brief       calculates the intersection end points with the object
     * \param[in]   p: pointer to 3-element array of the x-ray source position
     * \param[in]   r: pointer to 3-element array of the x-ray trajectory
     * \param[in]  ts: 2-element array to store the interaction positions
     * \return      true if successful, false otherwise
     */
    bool intersectionEndPoints(double* p, double* r, double* ts);

    /**
     * \fn          intersectionEndPoints_centeredAndNormalized
     * \brief       same as intersectionEndPoints, except the p and r values have been
     *              altered such that the object is now centered and has unit radii
     * \param[in]   p: pointer to 3-element array of the x-ray source position
     * \param[in]   r: pointer to 3-element array of the x-ray trajectory
     * \param[in]  ts: 2-element array to store the interaction positions
     * \return      true if successful, false otherwise
     */
    bool intersectionEndPoints_centeredAndNormalized(double* p, double* r, double* ts);

    /**
     * \fn          intersectionEndPoints_centeredAndNormalized
     * \brief       calculates the intesection with [-1, 1]
     * \param[in]   ts: 2-element array to store the interaction positions
     * \param[in]   p: pointer to 3-element array of the x-ray source position
     * \param[in]   r: pointer to 3-element array of the x-ray trajectory
     * \param[in]  ts: 2-element array to store the interaction positions
     * \return      true if successful, false otherwise
     */
    bool parametersOfIntersection_1D(double* ts, double p, double r);

    /**
     * \fn          parametersOfClippingPlaneIntersections
     * \brief       modifies the intersections to handle the intersection planes
     * \param[in]   ts: 2-element array to store the interaction positions
     * \param[in]   p: pointer to 3-element array of the x-ray source position
     * \param[in]   r: pointer to 3-element array of the x-ray trajectory
     * \param[in]  ts: 2-element array to store the interaction positions
     * \return      true if successful, false otherwise
     */
    bool parametersOfClippingPlaneIntersections(double* ts, double* p, double* r);

    int type;
    float centers[3];
    float radii[3];
    float val;
    float A[9];
    float clip[3];
    float clippingPlanes[6][4];
    string chemForm;
    
    bool isRotated;
    int numClippingPlanes;
    float clipCone[2];

    void restore_cone_params();
    void scale_saved_cone_params(float scale_x, float scale_y, float scale_z);
    void printParameters();

private:
    double dot(double* x, double* y, int N = 3);
    float centers_save[3];
    float radii_save[3];
};

class meshObject
{
public:
    meshObject();
    ~meshObject();

    /**
     * \fn          reset
     * \brief       deletes all data
     */
    void reset();

    /**
     * \fn          init
     * \brief       sets all member variables
     * \param[in]   trianges: pointer to array of all triangles vertices
     * \param[in]   N: number of triangles
     * \param[in]   val: the density (or arbitrary value) to proscribe inside the meshed surface
     * \param[in]   chemForm: (optional) chemical formula of this material
     * \return      true if successful, false otherwise
     */
    bool init(float* triangles, int N, float val, const char* chemForm);

    float val;
    string chemForm;
    float* triangles;
    int numTriangles;
};

class phantom
{
public:

    // Constructor and destructor; these do nothing
    phantom();
    phantom(parameters*);
    ~phantom();

    /**
     * \fn          addObject
     * \brief       Changes the voxels values to a specified value inside a 3D geometric object
                    (any voxel values inside the object before this function are ignored, i.e., this function does not accumulate the values, it replaces them)
     * \param[in]   f, pointer to the volume data (on the CPU)
     * \param[in]   params, pointer to an instance of the parameters class
     * \param[in]   type, the enumerated object type; see enum objectType_list
     * \param[in]   c, pointer to a three-element array of the (x,y,z) coordinates of the center of the object
     * \param[in]   r, pointer to a three-element array of the (x,y,z) coordinates of the radii of the object
     * \param[in]   val, value of the voxels inside the object
     * \param[in]   A, pointer to a 3X3 rotation matrix of the object
     * \param[in]   clip, pointer to a three-element array that specifies the clipping planes along the (x,y,z) coordinates
     * \param[in]   oversampling, voxels are broken up into oversampling X oversampling X oversampling subvoxels to model the partial volume effect on voxels on the edge of the object
     * \return      returns true if all input are valid and CT volume parameter values are defined and valid, false otherwise
     */
    bool addObject(float* f, parameters* params, int type, float* c, float* r, float val, float* A = NULL, float* clip = NULL, const char* chemForm = NULL, int oversampling = 1);

    /**
     * \fn          addObject
     * \brief       Changes the voxels values to a specified value inside a 3D geometric object
                    (any voxel values inside the object before this function are ignored, i.e., this function does not accumulate the values, it replaces them)
     * \param[in]   type, the enumerated object type; see enum objectType_list
     * \param[in]   c, pointer to a three-element array of the (x,y,z) coordinates of the center of the object
     * \param[in]   r, pointer to a three-element array of the (x,y,z) coordinates of the radii of the object
     * \param[in]   val, value of the voxels inside the object
     * \param[in]   A, pointer to a 3X3 rotation matrix of the object
     * \param[in]   clip, pointer to a three-element array that specifies the clipping planes along the (x,y,z) coordinates
     * \return      returns true if all input are valid and CT volume parameter values are defined and valid, false otherwise
     */
    bool addObject(int type, float* c, float* r, float val, float* A = NULL, float* clip = NULL, const char* chemForm = NULL);

    /**
     * \fn          addMesh
     * \brief       adds a mesh-based object to the stack
     * \param[in]   triangles: array of the vertices of all triangles in the mesh
     * \param[in]   numTriangles: the number of triangles in the mesh
     * \param[in]   val: the density (or arbitrary value) to proscribe inside the meshed surface
     * \param[in]   chemForm: (optional) chemical formula of this material
     * \return      returns true if all input are valid and CT volume parameter values are defined and valid, false otherwise
     */
    bool addMesh(float* triangles, int numTriangles, float val, const char* chemForm = NULL);

    /**
     * \fn          clearObjects
     * \brief       clears all geometic solids from the stack
     */
    void clearObjects();

    /**
     * \fn          clearMeshes
     * \brief       clears all meshes from the stack
     */
    void clearMeshes();

    /**
     * \fn          clearAll
     * \brief       clears all class members
     */
    void clearAll();

    /**
     * \fn          lineIntegral
     * \brief       calculates the line integral through the phantom
     * \param[in]   p: pointer to 3-element array of the x-ray source position
     * \param[in]   r: pointer to 3-element array of the x-ray trajectory
     * \return      line integral value
     */
    double lineIntegral(double* p, double* r);

    /**
     * \fn          synthesizeSymmetry
     * \brief       converts a 2D cylindrically-symmetric image into a 3D volume
     * \param[in]   f_radial: pointer to 2D cylindrically-symmetric image (input)
     * \param[in]   f: pointer to 3D volume (output)
     * \return      true if successful, false otherwise
     */
    bool synthesizeSymmetry(float* f_radial, float* f);

    /**
     * \fn          scale_phantom
     * \brief       scales the size of a geometric phantom
     * \param[in]   scale_x: magnification factor along x-coordinate
     * \param[in]   scale_y: magnification factor along y-coordinate
     * \param[in]   scale_z: magnification factor along z-coordinate
     * \return      true if successful, false otherwise
     */
    bool scale_phantom(float scale_x, float scale_y, float scale_z);

    /**
     * \fn          shift_phantom
     * \brief       shifts the position of a geometric phantom
     * \param[in]   shift_x: shifts along x-coordinate
     * \param[in]   shift_y: shifts along y-coordinate
     * \param[in]   shift_z: shifts along z-coordinate
     * \return      true if successful, false otherwise
     */
    bool shift_phantom(float shift_x, float shift_y, float shift_z);

    /**
     * \fn          voxelize
     * \brief       voxelizes a phantom composed of geometric solids
     * \param[in]   f: pointer to 3D volume (output)
     * \param[in]   params_in: pointer to a parameters object which describes the CT volume
     * \param[in]   oversampling: the over-sampling factor (a value great than one models partial volume)
     * \return      true if successful, false otherwise
     */
    bool voxelize(float* f, parameters* params_in, int oversampling = 1);

    /**
     * \fn          double_cone
     * \brief       creates a double-cone indicator function to be used for cone-beam artifact mitigation methods
     * \param[in]   f: pointer to 3D volume (input and output)
     * \param[in]   N_1: number of elements in the first dimension
     * \param[in]   N_2: number of elements in the second dimension
     * \param[in]   N_3: number of elements in the third dimension
     * \param[in]   beta: the aperture angle (in degrees) of the cone
     * \param[in]   minimum_radius: the minimum radius to consider as part of the double cone
     * \return      true if successful, false otherwise
     */
    bool double_cone(float* f, int N_1, int N_2, int N_3, float beta, float minimum_radius);

    void printAll();

    // enumerated list of all the 3D geometric shapes that are supported
    enum objectType_list { ELLIPSOID = 0, PARALLELEPIPED = 1, CYLINDER_X = 2, CYLINDER_Y = 3, CYLINDER_Z = 4, CONE_X = 5, CONE_Y = 6, CONE_Z = 7 };

    // Stack of the geometric solids-based phantom
    std::vector<geometricObject> objects;

    // Stack of the geometric mesh-based phantom
    std::vector<meshObject*> meshes;

    // List of all the unique materials in the phantom
    std::vector<string> materialTypes;

    /**
     * \fn          voxelize
     * \brief       returns the unique material type index
     * \param[in]   n: the n-th material in the stack
     * \return      the unique material type index
     */
    int getMaterialType(int n);

    /**
     * \fn          makeTempData
     * \brief       allocates the intData and floatData arrays (private member variables)
     * \param[in]   num_threads: number of cpu threads
     * \return      true if successful, false otherwise
     */
    bool makeTempData(int num_threads);

    /**
	 * \fn          assign
	 * \brief       makes a deep copy of the given parameter object
	 */
	void assign(phantom& other);

private:

    /**
     * \fn          x_inv
     * \return      returns (x_val - x_0) / T_x
     */
    float x_inv(float x_val);

    /**
     * \fn          y_inv
     * \return      returns (y_val - y_0) / T_y
     */
    float y_inv(float y_val);

    /**
     * \fn          z_inv
     * \return      returns (z_val - z_0) / T_z
     */
    float z_inv(float z_val);

    float clipCone[2];

    float x_0; // copy of parameters::x_0()
    float y_0; // copy of parameters::y_0()
    float z_0; // copy of parameters::z_0()

    int numX; // copy of parameters::numX
    int numY; // copy of parameters::numY
    int numZ; // copy of parameters::numZ
    float T_x; // copy of parameters::voxelWidth
    float T_y; // copy of parameters::voxelWidth
    float T_z; // copy of parameters::voxelHeight

    /**
     * \fn          isInside
     * \brief       This function is called by addObject, where in addObject, the coordinate system is shifted so the object is centered on the origin,
     *              scaled, so that all the object axes are 1.0, and rotated.  Then this function is called to test whether the given location is inside
     *              or outside the given shape.
     * \param[in]   x, x-coordinate
     * \param[in]   y, y-coordinate
     * \param[in]   z, z-coordinate
     * \param[in]   type, the enumerated object type; see enum objectType_list
     * \param[in]   clip, pointer to a three-element array that specifies the clipping planes along the (x,y,z) coordinates
     * \return      returns true the given (x,y,z) coordinates are inside the the shifted, rotated, and normalized 3D geometric shape
     */
    bool isInside(float x, float y, float z, int type, float* clip);

    int* intData;
    double* floatData;

    // local copy of the pointer to the parameters class that is passed by the addObject function
    parameters* params;
};

#endif
