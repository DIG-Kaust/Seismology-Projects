![Seismology](https://github.com/DIG-Kaust/Seismology-Projects/blob/main/logo.png)

A collection of projects performed by my students in the ErSE 210 Seismology course. 

All projects have been carried out on the open [Volve dataset](https://www.equinor.com/energy/volve-data-sharing). Our ambition is to apply an extensive set of processing techniques to the dataset during the years and provide a one-stop-shop for instructors and researchers willing to use the Volve dataset for their teaching or researcher.


## Projects

### 2021

- Juan Daniel Romero [jdromerom](https://github.com/jdromerom): [Wavelet estimation](jromero_waveletestimation)

### 2022

- Danilo Chamorro Riascos [dchamorror](https://github.com/dchamorror): [Semblance analysis and NMO correction](dchamorro_nmo)

- Ning Wang [WANGN0E](https://github.com/WANGN0E): [Up/Down wavefield separation](nwang_wavsep)

- Eyad Babtain: [Kirchhoff Depth Migration](ebabtain_kirchhoff)


### 2023

- Arturo Ruiz [arturoruizs](https://github.com/arturoruizs): [3D Deblending](aruiz_3ddebl)

- Muhammad Iqbal Khatami [iqbalkhatami16](https://github.com/iqbalkhatami16): [Multiple removal using parabolic radon transform](ikhatami_radondem)
 
- Xiao Ma: [Kirchoff Time Migration](xma_kirchoofftime)


### 2024

- Emerald Olango: [Prestack Kirchoff Time Migration](eolango_prestackkirchoofftime)

- Sophia Manzo Vega: [Dead trace identification](smanzo_deadtraces)

- Amnah Samarin - Ulises Berman [Amnah2020](https://github.com/Amnah2020)-[uber30](https://github.com/uber30): [Refraction tomography](https://github.com/Amnah2020/RefrTomo/tree/Volvo)

- Charbel Sayan: [Predictive Decon](csayan_predictivedecon)

- Cristhian Valladares: [Time to Depth Conversion](cvalladares_time2depth)
 
 
## Data

In most cases, our projects start from one of the openly available data in the Volve village. In this case, we refer to this [notebook](https://github.com/PyLops/pylops_notebooks/blob/master/developement/SeismicInversion-Volve.ipynb) where the entire procedure to download data from the Volve data village is explained in details.

We also provide a script in the `data_preparation` directory, which generates two subsets of the main dataset used in some of the projects.

Finally, in some cases we used derived datasets from other projects of ours or available on the web. In this case, we provide details on how to obtain such data in the project folder directly.


## Contribute

Note that these projects are the results of a few weeks of work and therefore may be unfinished; nevertheless, they provide good starting points for whoever interested to work with the Volve dataset at different stages of processing.

Contributions to improve any of the projects are most welcome! Please simply open a GitHub issue if you find any bug or want to add any feature to the current material. We will be happy to discuss with you and work together on making a PR at any time.