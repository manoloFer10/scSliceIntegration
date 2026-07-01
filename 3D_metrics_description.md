# 3D Per-Object Metrics Reference

This document describes every metric extracted per segmented object from the
NIfTI volumes. Metrics are grouped by source: **shape** (derived from the
segmentation label map alone) and **fluorescence intensity** (derived from the
green channel image, masked per object). A third group covers **derived
metrics** computed by combining the two.

All physical metrics are expressed in the spatial units stored in the NIfTI
header — typically millimetres (mm) for clinical imaging or micrometres (µm)
for microscopy. **Verify the header units before interpreting absolute values.**

---

## 1. Object identification

| Column | Description |
|---|---|
| `sample` | Name of the subfolder (replicate) this object belongs to. |
| `label` | Integer label assigned to this object after connected-component labeling. Labels are assigned in descending order of object size (label 1 = largest object). |
| `n_voxels` | Raw voxel count of the object. Use this as a sanity check: objects with very few voxels (< 10) are likely segmentation noise and should be filtered before analysis. |

---

## 2. Shape metrics
*Source: `LabelShapeStatisticsImageFilter` applied to the segmentation label map.*
*All linear/area/volume values are in physical (calibrated) units from the NIfTI header.*

---

### 2.1 Size

#### `volume_phys`
Total physical volume of the object.

$$V = N_{\text{voxels}} \times v_x \times v_y \times v_z$$

where $v_x, v_y, v_z$ are the voxel spacings in each dimension.
**Units:** spatial\_unit³ (e.g. µm³).
Correlates with cell size and, in proliferating populations, with cell cycle phase.

#### `surface_area_phys`
Surface area of the object boundary, computed by SimpleITK from the label map
boundary voxels using a discrete surface estimation.
**Units:** spatial\_unit² (e.g. µm²).
Relevant for membrane-dependent processes (signal exchange, receptor density).

#### `equiv_spherical_radius`
Radius of a sphere that would have the same volume as this object.

$$r_{\text{equiv}} = \left(\frac{3V}{4\pi}\right)^{1/3}$$

**Units:** spatial\_unit (e.g. µm).
A size descriptor that is independent of object shape, useful for comparing
objects of different morphologies on a common scale.

#### `surface_to_volume_ratio`
Ratio of surface area to volume.

$$\text{SVR} = \frac{A}{V}$$

**Units:** spatial\_unit⁻¹.
For a perfect sphere, SVR = 3/r. Larger values indicate more irregular or
elongated objects relative to their size. Biologically relevant for diffusion
and membrane-to-cytoplasm ratios.

---

### 2.2 Sphericity and roundness

#### `sphericity`
How closely the object's surface area matches that of a sphere of equal volume.
A sphere has the minimum possible surface area for a given volume.

$$\Psi = \frac{\pi^{1/3}(6V)^{2/3}}{A}$$

**Range:** (0, 1]. Value of 1 = perfect sphere; values approaching 0 indicate
highly irregular or elongated shapes.
This is the primary shape regularity descriptor. It is sensitive to surface
roughness, protrusions, and irregular segmentation boundaries.

#### `roundness`
SimpleITK's built-in roundness metric. Defined as the ratio of the surface
area of the equivalent sphere to the actual surface area.

$$\text{Roundness} = \frac{A_{\text{sphere}}(r_{\text{equiv}})}{A_{\text{actual}}}$$

**Range:** (0, 1]. Value of 1 = perfect sphere.
Numerically similar to sphericity but derived differently. Reporting both
and checking their correlation serves as a cross-validation: large divergence
between the two on the same object is a red flag for segmentation artifacts
(e.g. spiky or fragmented surfaces from noisy masks).

---

### 2.3 Axis-based shape descriptors

These metrics are derived from the **principal axes of the inertia tensor** of
the object — the three orthogonal directions along which mass (voxel count) is
maximally distributed. The three axis lengths are ordered $d_1 \geq d_2 \geq d_3$.

#### `elongation`
Ratio of the second to the first principal axis length.

$$\text{Elongation} = \frac{d_2}{d_1} \leq 1$$

**Range:** (0, 1]. Value of 1 = no elongation (isotropic in the two major
axes). Values approaching 0 indicate a rod- or needle-like shape.
Captures cell polarisation and migratory morphology.

#### `flatness`
Ratio of the third to the second principal axis length.

$$\text{Flatness} = \frac{d_3}{d_2} \leq 1$$

**Range:** (0, 1]. Value of 1 = not flat. Values approaching 0 indicate a
disc- or pancake-like shape.
Particularly informative for adherent cells, which are typically flat in the
z-axis relative to their xy spread.

> **Reading elongation and flatness together:**
> A sphere has elongation ≈ 1, flatness ≈ 1.
> A rod has elongation < 1, flatness ≈ 1.
> A disc has elongation ≈ 1, flatness < 1.
> An irregular blob will show both < 1.

---

### 2.4 Spatial position

#### `centroid_x`, `centroid_y`, `centroid_z`
Geometric centroid of the object in physical (world) coordinates, as defined
by the NIfTI affine.
**Units:** spatial\_unit.
Used for spatial distribution analysis across replicates (e.g. clustering,
distance to image boundary, relative positioning between objects).

---

## 3. Fluorescence intensity metrics
*Source: `LabelIntensityStatisticsImageFilter` applied to the green fluorescence*
*channel image, using the segmentation label map as the object mask.*
*Each metric is computed only over the voxels belonging to a given object.*

---

### 3.1 Central tendency

#### `fluor_mean`
Arithmetic mean intensity across all voxels of the object.
Sensitive to bright outlier voxels (e.g. saturation artefacts or bright foci).

#### `fluor_median`
Median intensity across all voxels of the object.
More robust than the mean when the intensity distribution is skewed or contains
outlier voxels.

---

### 3.2 Spread and distribution shape

#### `fluor_std`
Standard deviation of intensity across the object's voxels.
High values indicate heterogeneous signal (e.g. bright foci on a dim
background); low values indicate uniform signal.

#### `fluor_min` / `fluor_max`
Minimum and maximum voxel intensity within the object.
`fluor_max` near the sensor's dynamic range ceiling suggests potential
saturation; `fluor_min` near zero suggests the object boundary may include
background voxels.

#### `fluor_skewness`
Third standardised moment of the within-object intensity distribution.

$$\text{Skewness} = \frac{\mathbb{E}[(I - \mu)^3]}{\sigma^3}$$

Positive skewness: most voxels are dim with a tail of bright voxels (e.g.
a few bright puncta). Negative skewness: most voxels are bright.
Useful for distinguishing diffuse vs. punctate signal patterns.

#### `fluor_kurtosis`
Fourth standardised moment of the intensity distribution.

$$\text{Kurtosis} = \frac{\mathbb{E}[(I - \mu)^4]}{\sigma^4}$$

High kurtosis (leptokurtic) indicates a sharp central peak with heavy tails —
characteristic of bright foci on a near-zero background. Low kurtosis
(platykurtic) indicates a flat, broad distribution.

---

### 3.3 Total signal

#### `fluor_sum`
Sum of all voxel intensities within the object (integrated intensity).

$$\text{Sum} = \sum_{i \in \text{object}} I_i$$

In quantitative fluorescence microscopy, integrated intensity is a proxy for
total amount of labelled molecule within the object, assuming linear response
and flat-field corrected images. This is the basis of DNA content cell-cycle
staging by DAPI sum intensity.
**Units:** intensity × voxels (dimensionless count if intensity is in ADU).

---

### 3.4 Spatial distribution of fluorescence

#### `fluor_weighted_cx`, `fluor_weighted_cy`, `fluor_weighted_cz`
Intensity-weighted centroid of the object: the centre of fluorescence mass,
where each voxel's contribution is proportional to its intensity.

$$\vec{c}_w = \frac{\sum_{i} I_i \vec{x}_i}{\sum_{i} I_i}$$

**Units:** spatial\_unit.
If fluorescence is uniformly distributed, the weighted centroid coincides
with the geometric centroid. Displacement between the two indicates asymmetric
signal distribution within the object.

---

## 4. Derived metrics
*Computed from the shape and intensity metrics above after merging.*

---

#### `centroid_displacement`
Euclidean distance between the geometric centroid and the fluorescence-weighted
centroid.

$$d = \left\| \vec{c}_{\text{geom}} - \vec{c}_{\text{fluor}} \right\|_2$$

**Units:** spatial\_unit.
A non-zero displacement indicates that fluorescent signal is concentrated
asymmetrically within the object — a polarity indicator. Large values may also
reflect segmentation inaccuracy (label boundary does not match signal boundary).

#### `fluor_cv`
Coefficient of variation of intensity within the object.

$$\text{CV} = \frac{\sigma}{\mu}$$

**Range:** [0, ∞). Dimensionless.
Normalises within-object heterogeneity by the mean, allowing comparison across
objects of different brightness levels. High CV = heterogeneous signal; low CV
= uniform signal.

#### `fluor_density`
Integrated intensity normalised by object volume.

$$\text{Density} = \frac{\text{Sum}}{V}$$

**Units:** intensity / spatial\_unit³.
Equivalent to mean intensity but with explicit volume normalisation. Useful
when comparing total signal between objects of different sizes while controlling
for size effects.

---

## 5. Metrics at a glance

| Group | Column | Biologically informative for |
|---|---|---|
| ID | `sample`, `label`, `n_voxels` | Bookkeeping; noise filtering |
| Size | `volume_phys`, `equiv_spherical_radius` | Cell size, cell cycle staging |
| Surface | `surface_area_phys`, `surface_to_volume_ratio` | Membrane area, diffusion |
| Regularity | `sphericity`, `roundness` | Morphological regularity, segmentation QA |
| Axes | `elongation`, `flatness` | Polarisation, adherence, migration |
| Position | `centroid_x/y/z` | Spatial organisation |
| Intensity | `fluor_mean`, `fluor_median` | Signal level per cell |
| Spread | `fluor_std`, `fluor_min/max`, `fluor_skewness`, `fluor_kurtosis` | Signal heterogeneity, puncta vs. diffuse |
| Total signal | `fluor_sum` | Total protein/molecule proxy |
| Signal location | `fluor_weighted_cx/y/z` | Fluorescence asymmetry |
| Derived | `centroid_displacement` | Signal polarity |
| Derived | `fluor_cv` | Normalised heterogeneity |
| Derived | `fluor_density` | Size-normalised signal level |

---

## 6. Important caveats

- **Header units must be verified.** All physical metrics scale with voxel
  spacing. If the NIfTI header stores spacing in mm but the data are in µm,
  volumes will be off by 10⁹.
- **`fluor_sum` requires flat-field correction** to be a valid proxy for total
  molecule count. Without it, it reflects illumination unevenness as much as
  biology.
- **Sphericity is sensitive to segmentation surface quality.** Rough or noisy
  label boundaries inflate surface area and deflate sphericity. Apply a mild
  label smoothing step if sphericity values appear systematically low.
- **Objects with < 10 voxels** should be treated as noise until validated
  visually. Their surface area estimates are unreliable.
- **`centroid_displacement`** is only interpretable if the fluorescence image
  and label map are in the same physical space (verified by the spacing check
  at load time).