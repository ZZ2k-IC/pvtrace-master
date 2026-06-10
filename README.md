> Optical ray tracing for room-temperature maser gain media and invasive optical pumping structures

# Ray-tracing optical pumping systems for solid-state masers

*pvtrace-maser* is a modified version of *pvtrace*, a statistical photon path tracer written in Python.

Rays are propagated through a 3D scene and their interactions with materials, interfaces, and embedded optical structures are recorded to build up statistical information about optical power transport and absorption.

This project is based on:

- Daniel Farrell's original **pvtrace**
- Shomik Verma's **pvtrace-sv**

The current version further extends the framework for modelling optical pumping in room-temperature solid-state masers, particularly pentacene-doped para-terphenyl (Pc:PTP) systems.

Unlike the original pvtrace package, which primarily targets luminescent solar concentrators (LSCs), this version focuses on:

- invasive optical pumping
- embedded optical waveguides
- multi-part optical injectors
- optical absorption distribution
- maser gain-medium illumination

---

# Development History

## Original pvtrace

Original repository:

https://github.com/danieljfarrell/pvtrace

A Monte-Carlo optical ray-tracing package for luminescent materials, luminescent solar concentrators (LSCs), and spectral conversion devices.

## pvtrace-sv

Modifications introduced by Shomik Verma:

- Support for unconventional LSC geometries
- Surface normal recording during ray tracing
- Parallel simulation scripts
- Initial PySide2 GUI support

Original repository:

https://github.com/shomikverma/pvtrace-sv

## Current Extension

Additional modifications introduced for room-temperature maser research:

### Waveguide–Gain-Medium Architecture

Support for simulations containing:

- Gain medium
- Embedded optical waveguide / injector

allowing direct simulation of invasive optical pumping structures.

### Multi-Part STL Waveguide Support

Support for importing multiple STL files as a single optical injector structure.

Applications include:

- wedge injectors
- multi-blade injectors
- arbitrary embedded waveguide geometries

### Maser-Specific Absorption Model

This version is designed specifically for Pc:PTP maser pumping simulations.

Key assumptions include:

- absorption evaluated along the crystal pumping axis (Z-axis)
- Beer-Lambert absorption based on projected propagation distance
- optical penetration modelling for embedded waveguide systems

### Ray-Tracing Engine Improvements

Several kernel-level modifications have been implemented, including:

- robust container identification for multi-part geometries
- improved STL overlap handling
- corrected optical path-length accounting
- improved waveguide-to-crystal transport
- improved interface handling

### Analysis Features

- absorption heatmaps
- detector-plane support
- custom LED angular emission distributions

---

# GUI

A PySide2-based graphical user interface is included.

Run:

```bash
python GUI/main.py
```

The GUI supports:

- gain-medium definition
- embedded waveguide definition
- STL geometry import
- optical source configuration
- detector placement
- uniformity analysis
- result export

---

# Installation

Tested using:

```text
Python 3.7.9 (3.7.0 < Any versions < 3.8.0)
```

Clone the repository:

```bash
git clone https://github.com/ZZ2k-IC/pvtrace-maser.git
cd pvtrace-maser
```

Install locally:

```bash
pip install -e .
```

Required packages:

```text
numpy
pandas
matplotlib
anytree
meshcat>=0.0.16
trimesh[easy]
PySide2
progressbar2
```

All required packages should be installed automatically through `setup.py`.

---

# Acknowledgements

- Daniel Farrell — original pvtrace framework
- Shomik Verma — pvtrace-sv modifications
- Imperial College London Mark Oxborrow Maser Group

---

# Disclaimer

This fork is not intended as a general-purpose luminescent solar concentrator simulator.

It has been specifically developed for optical pumping simulations of room-temperature solid-state masers and embedded waveguide structures.