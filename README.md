# Fusing Multisensory Signals Across Channels and Time

This repository contains the code and data associated with our preprint "Fusing Multisensory Signals Across Channels and Time" (Anil, Ghosh & Goodman, 2024). The work investigates how temporal structure in multisensory signals can be leveraged in integration strategies, introducing novel computational models for analyzing sensory fusion across multiple timescales.

## Overview

We extend previous work on multisensory integration by examining how temporal dependencies affect signal processing strategies. The repository includes implementations of:

- Time-dependent detection tasks with variable burst lengths
- Linear and nonlinear fusion algorithms (LF, NLF)
- Sliding window integration models (NLFw)
- Recurrent neural network architectures
- Lévy flight-based signal generation

## Dependencies

A complete environment specification is provided in `environment.yml`. Key dependencies include:
- NumPy
- PyTorch
- scikit-learn
- Pandas
- Matplotlib

## Reproduction

To reproduce the main results from our paper:

1. Install dependencies: `conda env create -f environment.yml`
2. Run experiment scripts in `scripts/`
3. Generate figures using notebooks in `Plotter/`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about the code or paper, please contact:
- Dan F.M. Goodman (d.goodman@imperial.ac.uk)