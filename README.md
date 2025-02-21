# Vision-Language Integration for Domain Generalisation

This repository contains the implementation of two frameworks proposed in my Master's thesis at the University of Auckland: Domain Description-Guided Experts (ODE) and Meta-Feature Adapter (MFA). These frameworks leverage vision-language models to address domain generalisation challenges in both conventional and heterogeneous scenarios.

## Overview

### Domain Description-Guided Experts (ODE)
ODE improves model generalisation ability by leveraging domain descriptions to guide both the learning and utilisation of domain-specific knowledge. The framework effectively captures and preserves domain-specific knowledge through domain expert adapters while maintaining domain-invariant representations in a shared backbone. During inference, domain descriptions guide the integration of domain-specific knowledge through a similarity-based mechanism.

### Meta-Feature Adapter (MFA)
MFA is designed to improve animal re-identification performance by integrating environmental metadata with visual features. The framework translates environmental metadata (temperature, circadian rhythms, face orientation) into textual descriptions and incorporates them through a gated cross-attention mechanism. We also introduce the Metadata Augmented Animal Re-identification (MAAR) dataset to evaluate our approach.

