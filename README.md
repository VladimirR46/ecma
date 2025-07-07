# ecma
ECMA (Extended Clustering and Model Approximation) is an algorithm designed for the detection of precise oculomotor features from eye-tracking signals under high-noise conditions.

The algorithm utilizes k-means clustering within a sliding time window, incorporating time delays to enhance the temporal resolution. Following the clustering step, the detected segments are approximated using a parametric saccade model, allowing for accurate characterization of eye movement dynamics despite elevated noise levels.

Installation & Usage
====================

To install that package, clone this git, open a terminal on the root of the git folder and type:
```bash
pip install .
```

Or, without cloning, simply run the following command
```bash
pip install git+https://github.com/VladimirR46/ecma.git
```

