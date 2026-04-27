# Measuring Alignment Friction: A Gradient Decomposition Assay

This repository contains the paper, telemetry dataset, and origin documents for the Gradient Decomposition Assay (GDA). The GDA evaluates how frontier language models shift from high-density systems analysis to procedural scaffolding when subjected to institutional constraints and adversarial pressure.

### Core Finding
Direct constraints cause models to generate procedural boilerplate, but placing the same structural analysis inside a fictional/counterfactual wrapper (Counterfactual Narrative Reframing) recovers analytical density and drastically reduces visible safety friction.

### Repository Contents
* **`GDA_Alignment_Friction_Paper.pdf`**: The formal empirical study detailing the 8-vector prompt ladder and the 2,000-run telemetry across 5 model families.
* **`research_genealogy.md`**: A companion document detailing the philosophical origins of the assay, the "Bugs Bunny Optimization," and the inadvertent methodological hygiene created by context-window limitations.
* **Dataset**: The raw structured CSV telemetry (valid runs and flagged invalid runs) is hosted here and also available on Hugging Face.
