# Third-Party Notices

## D3

Repository: https://github.com/Zig-HS/D3  
Pinned commit: `c798fbc57fe0c4198d63a73732c2c0f9e4b4816c`  
License: MIT  
Paper: https://arxiv.org/abs/2508.00701

This project implements an optional local adapter for the D3 second-order temporal-feature computation with a documented single-video preprocessing adaptation. v0.8.1 documents mathematical parity for the score computation with synthetic tensors, but does not claim full upstream runtime equivalence until actual pretrained encoder inference and upstream runtime parity are executed in the local environment.

## Optional Model Libraries

Optional D3 inference may use PyTorch, torchvision, Transformers, and timm according to `requirements-d3.txt`. These packages and any pretrained model assets are not vendored in this repository.
