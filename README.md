# slugtracer(rtcamp9)
![](document/image/001.PNG)

[レイトレ合宿9](https://sites.google.com/view/rtcamp9/home)に提出したレンダラー

- [レンダラー紹介のスライド](https://speakerdeck.com/takashikachan/slugtracer-rtcamp09)

## Features
- Path Tracing 
- Disney Principled BRDF
- Correlated Multi Jittered Sampling
- An Analytic Model for Full Spectral SkyDome Radiance
- Polynomial Optics
- Optix Denoiser
- Cuda Kernel(Tonemapping)

## Requirements
- C++ 20
- CUDA 12.1
- OptiX 7.7
- CMake 3.15.0以上
- Windows10,11

## Build

下記をインストール
- CUDA Tool kit v12.1
    - https://developer.nvidia.com/cuda-12-1-0-download-archive
- Optix SDK 7.7
    - https://developer.nvidia.com/designworks/optix/downloads/legacy


下記コマンドを実行。
```
git submodule update --init
```

下記batファイルを実行
- SetupEnv.bat

workspaceフォルダが作られ、その中にソリューションが生成されます。

## References

- [Polynomial Optics: A Construction Kit for Efficient Ray-Tracing of Lens Systems](https://www.cs.ubc.ca/labs/imager/tr/2012/PolynomialOptics/)
- [Correlated Multi-Jittered Sampling](https://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf)
- [An Analytic Model for Full Spectral Sky-Dome Radiance](https://cgg.mff.cuni.cz/projects/SkylightModelling/HosekWilkie_SkylightModel_SIGGRAPH2012_Preprint_lowres.pdf)
