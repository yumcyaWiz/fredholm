# fredholm

Research oriented GPU offline renderer.

![](img/rtcamp8.png)
![](img/gui.jpg)

## Requirements

* C++17
* CUDA 11.7
* OptiX 7.5

## Features

* Path tracing integrator with multi importance sampling
* [Arnold Standard Surface](https://autodesk.github.io/standard-surface/)
* [Correlated Multi Jittered Sampling](https://graphics.pixar.com/library/MultiJitteredSampling/#:~:text=Abstract%3A,to%20which%20they%20are%20prone.)
* [Hosek Sky Model](https://cgg.mff.cuni.cz/projects/SkylightModelling/)
* `.obj`, `.gltf` scene file
* Post process kernels(Bloom, Chromatic aberration, Tone mapping)

## Build

```
git submodule update --init
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Run

```
cd build
./app/gui
```

## References

* [Arnold Standard Surface](https://autodesk.github.io/standard-surface/)
* [Estevez, A. C., & Kulla, C. (2017). Production Friendly Microfacet Sheen BRDF. ACM SIGGRAPH 2017.](http://www.aconty.com/pdf/s2017_pbs_imageworks_sheen.pdf)
* [Gulbrandsen, O. (2014). Artist friendly metallic fresnel. Journal of Computer Graphics Techniques, 3(4).](https://jcgt.org/published/0003/04/03/)
* [Heitz, E. (2018). Sampling the GGX distribution of visible normals. Journal of Computer Graphics Techniques (JCGT), 7(4), 1-13.](https://jcgt.org/published/0007/04/01/)
* [Hosek, L., & Wilkie, A. (2012). An analytic model for full spectral sky-dome radiance. ACM Transactions on Graphics (TOG), 31(4), 1-9.](https://cgg.mff.cuni.cz/projects/SkylightModelling/)
* [Kensler, A. (2013). Correlated multi-jittered sampling.](https://graphics.pixar.com/library/MultiJitteredSampling/#:~:text=Abstract%3A,to%20which%20they%20are%20prone.)
* https://www.shadertoy.com/view/wl2SDt
* https://www.shadertoy.com/view/llXyWr
* https://github.com/ingowald/optix7course