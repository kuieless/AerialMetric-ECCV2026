<div align="center">
  
# AerialMetric: Benchmarking and Adapting UAV Monocular Metric Depth Estimation in the Real World

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://kuieless.github.io/AerialMetric-ECCV2026-page/)
[![Dataset & Weights](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026)

**[Zhongqiang Song](https://kuieless.github.io/Kuie-s-Academic-Pages/)<sup>1</sup> &nbsp;&nbsp; [Guanying Chen](https://guanyingc.github.io/)<sup>1,✉</sup> &nbsp;&nbsp; [Yuqi Zhang](https://zyqz97.github.io/)<sup>2,3</sup> &nbsp;&nbsp; Yin Zou<sup>1</sup>**

**[Chuanyu Fu](https://fcyycf.github.io/)<sup>1</sup> &nbsp;&nbsp; Zhiyuan Yuan<sup>1</sup> &nbsp;&nbsp; [Chuan Huang](https://scholar.google.com/citations?user=ei4jR4IAAAAJ&hl=en)<sup>4,2</sup> &nbsp;&nbsp; [Shuguang Cui](https://scholar.google.com.hk/citations?user=1o_qvR0AAAAJ&hl=en)<sup>3,2</sup> &nbsp;&nbsp; [Xiaochun Cao](https://scholar.google.com/citations?user=PDgp6OkAAAAJ&hl=en)<sup>1</sup>**

<br>

<sup>1</sup> Sun Yat-sen University, Shenzhen Campus &nbsp;&nbsp;&nbsp;&nbsp;
<sup>2</sup> FNii-Shenzhen &nbsp;&nbsp;&nbsp;&nbsp;
<sup>3</sup> SSE, CUHKSZ &nbsp;&nbsp;&nbsp;&nbsp;
<sup>4</sup> SIAS, USTC

<p align="center">
  <img src="./static/images/teasernyu.jpg" width="95%">
</p>

<p align="center">
  <em>AerialMetric is a real-world UAV monocular metric depth benchmark covering oblique photogrammetry, controlled aerial variables, in-the-wild UAV videos, and adapted MoGe-2 baselines.</em>
</p>
</div>

## Overview

AerialMetric focuses on monocular metric depth estimation for real-world UAV scenarios. The benchmark covers diverse aerial conditions, including oblique photogrammetry, controlled aerial variables, in-the-wild UAV videos, and ground-domain evaluation for adapted MoGe-2 baselines.

## Datasets & Weights

<p align="center">
  <table>
    <tr>
      <td width="50%" align="center">
        <img src="static/images/figure-piplinefcy0304.jpg" width="100%">
      </td>
      <td width="50%" align="center">
        <img src="static/images/fig3dataset.jpg" width="100%">
      </td>
    </tr>
    <tr>
      <td align="center">
        <em>Pipeline of AerialMetric.</em>
      </td>
      <td align="center">
        <em>Dataset composition of AerialMetric.</em>
      </td>
    </tr>
  </table>
</p>

| Resource | Link |
|---|---|
| Dataset & Weights | [Kuiee/AerialMetric-ECCV2026](https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026) |

## Code Release

The code, benchmark scripts, and detailed reproduction instructions will be released after the public release schedule.

## Acknowledgments

### Models & Tools

- **[MoGe-2](https://github.com/microsoft/MoGe)** (Wang et al., 2025) - the core monocular geometry estimation model.

### Oblique Data Sources

We gratefully acknowledge the following datasets or works for providing raw UAV imagery and scene assets:

- **[GauU-Scene](https://saliteta.github.io/CUHKSZ_SMBU/)** - large-scale UAV aerial scene dataset.
- **[UrbanBIS](https://vcc.tech/UrbanBIS)** (Yang et al., SIGGRAPH 2023) - benchmark for fine-grained urban building instance segmentation.
- **[UrbanScene3D](https://vcc.tech/UrbanScene3D/)** (Lin et al., ECCV 2022) - large-scale urban scene dataset with high-resolution aerial images.
- **[UAVScenes](https://github.com/sijieaaa/UAVScenes)** (Wang et al., ICCV 2025) - multi-modal UAV dataset with frame-wise semantic annotations.
- **[OpenDroneMap](https://github.com/OpenDroneMap/ODM)** - open-source toolkit for generating maps, point clouds, 3D models, and DEMs from drone images.
- **[ArcGIS Drone2Map](https://www.esri.com/en-us/arcgis/products/arcgis-reality/products/arcgis-drone2map)** (ESRI) - create 3D products from drone imagery.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{song2026aerialmetric,
  title     = {AerialMetric: Benchmarking and Adapting UAV Monocular Metric Depth Estimation in the Real World},
  author    = {Zhongqiang Song and Guanying Chen and Yuqi Zhang and Yin Zou and Chuanyu Fu and Zhiyuan Yuan and Chuan Huang and Shuguang Cui and Xiaochun Cao},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
