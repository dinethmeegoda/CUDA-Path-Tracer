# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Dineth Meegoda
  - [LinkedIn](https://www.linkedin.com/in/dinethmeegoda/), [Personal Website](https://www.dinethmeegoda.com).
- Tested on: Windows 10 Pro, Ryzen 9 5900X 12 Core @ 3.7GHz 32GB, RTX 3070 8GB

## Summary

This project is a foray into learning CUDA, specifically for graphics programming applications. The goal of this Path Tracer was to take the existing graphics concepts I'm already familiar with, and try to use parallel programming on the GPU with separate kernel invocations to 'simulate' the steps in the graphics pipeline that graphics frameworks typically abstract away. In the end, I wanted to implement as many visually striking features as I could to increase the number of tools I have to create pretty pictures, while making performance boosts to keep the path tracer usable for complex scenes.

### Path Tracing

Path tracing is a rendering technique where we 'shoot' out rays of light from each pixel and follow it's path to figure out what the contribution of light is given to that pixel. The specific color and direction of a ray depends on the materials it intersects and its physical properties. The specific kind of Path Tracing done in this program uses the Monte-Carlo Method in which a random sample is selected and divided by the value of a probability density function to make up for the contribution. Since it's random, one sample isn't enough to obtain an accurate image, so we average samples and eventually get an image we're happy with.

For more on path tracing, check out [PBRT](https://pbr-book.org/4ed/contents), which really helped me out during this project.

### Features Implemented

- Visual Features:
  - [BSDFs/BRDFs](#bsdfsbrdfs)
  - [Ray Dispersion](#ray-dispersion)
  - [Stochastic Sampled Anti-Aliasing](#stochastic-sampled-anti-aliasing)
  - [glTF Mesh Loading & Texture Mapping](#gltf-mesh-loading--texture-mapping)
  - [Reinhard Operator & Gamma Correction](#reinhard-operator--gamma-correction)
  - [Environment Mapping](#environment-mapping)
  - [Proceudral Shapes](#procedural-shapes)
  - [Procedural Textures](#procedural-textures)
  - [Denoising](#denoising)
- Performance Features:
  - [Bounding Volume Heirarchy](#bounding-volume-hierarchy)
  - [Stream Compaction](#stream-compaction)
  - [Material Sorting](#material-sorting)

## Visual Features

### BSDFs/BRDFs

### Ray Dispersion

### Stochastic Sampled Anti-Aliasing

### glTF Mesh Loading & Texture Mapping

### Reinhard Operator & Gamma Correction

### Environment Mapping

### Procedural Shapes

### Procedural Textures

### Denoising

## Performance Features

### Bounding Volume Hierarchy

### Stream Compaction

### Material Sorting

## Bloopers

## References
