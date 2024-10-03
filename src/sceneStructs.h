#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
};

struct Triangle {
    Vertex v1;
    Vertex v2;
    Vertex v3;
    glm::vec3 centroid;
};

struct Texture {
    int id;
    int width;
    int height;
    int numChannels;
    int startIndex;
	int endIndex;
};

struct Geom
{
    enum GeomType type;
    int meshid;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
	int triangleStart;
	int triangleEnd;
    int textureStart = -1;
	bool usesTexture = false;
    bool usesNormals = false;
    bool usesUVs = false;
};

struct Material
{
    glm::vec3 color;
    int hasReflective;
    int hasRefractive;
	int hasPlastic;
    float indexOfRefraction;
    float emittance;
	float roughness;
    float dispersion;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
	int waveLength;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  bool hasUV = false;
  glm::vec2 uv;
  int texid = -1;
  int materialId;
};

struct bbox {
    bbox() : bmin(1e30f), bmax(-1e30f) {}

    glm::vec3 bmin, bmax;
    __host__ __device__ void grow(glm::vec3 p) {
        bmin = glm::vec3{ glm::min(bmin.x, p.x), glm::min(bmin.y, p.y), glm::min(bmin.z, p.z) };
        bmax = glm::vec3{ glm::max(bmax.x, p.x), glm::max(bmax.y, p.y), glm::max(bmax.z, p.z) };
    }
    __host__ __device__ float area()
    {
        glm::vec3 e = bmax - bmin; // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

struct BVHNode {

	BVHNode() : aabb(), leftFirst(-1), numTriangles(0) {}

    bbox aabb;
	int leftFirst, numTriangles;
	__host__ __device__ bool isLeaf() { return numTriangles > 0; }
};