# Do Vision-Language Models Actually Understand Images?

### A Reproducible Benchmark for Visual Question Answering, Object Counting, Spatial Reasoning, Caption Alignment & Consistency

## Overview

Vision-Language Models (VLMs) have demonstrated impressive capabilities in image captioning, visual question answering, and multimodal understanding. However, accurately describing an image does not necessarily imply that a model understands its objects, spatial relationships, or underlying visual composition.

**Can vision-language models genuinely reason about visual information, or do they rely on statistical patterns and learned shortcuts?**

This project investigates that question through a reproducible evaluation framework that separates visual understanding into multiple measurable capabilities rather than reducing performance to a single benchmark score.

The benchmark evaluates:

- **Visual Question Answering:** Identifying visual attributes and answering image-grounded questions.
- **Object Counting:** Determining the exact number of target objects.
- **Spatial Reasoning:** Understanding left/right and above/below relationships.
- **Image–Text Alignment:** Distinguishing accurate captions from carefully designed counterfactual descriptions.
- **Reasoning Consistency:** Testing whether logically related questions produce consistent and correct answers.

The framework combines controlled synthetic scenes, pretrained vision-language models, statistical evaluation, and an optional real-world image extension.

## Research Questions

The notebook investigates five central questions:

1. Can a model answer basic visual questions while struggling with counting or spatial relationships?
2. How reliably can VLMs identify object quantities as scene density increases?
3. Do models distinguish genuine spatial relationships from familiar positional patterns?
4. Can image–text similarity models detect subtle errors in otherwise plausible captions?
5. Does consistent answering indicate correct visual reasoning, or can models consistently make mistakes?

**Central Hypothesis:** Strong performance on general image descriptions does not necessarily imply reliable object counting, relational reasoning, or visual grounding.

This is treated as a hypothesis to evaluate rather than a predetermined conclusion.

## Models Evaluated

| Model | Architecture / Interface | Evaluation |
|---|---|---|
| Salesforce BLIP VQA Base | Generative Visual Question Answering | VQA, Counting, Spatial Reasoning, Consistency |
| OpenAI CLIP ViT-B/32 | Contrastive Image–Text Matching | Caption Ranking and Counterfactual Alignment |

**Important:** BLIP and CLIP perform different tasks. Their results are evaluated independently rather than interpreted as a direct head-to-head comparison.

## Benchmark Architecture

```text
                 Visual Understanding Benchmark
                              |
                 Controlled Scene Generation
                              |
                 Ground-Truth Annotations
                              |
              +---------------+---------------+
              |                               |
          BLIP VQA                        CLIP Model
              |                               |
       +------+------+                 Caption Ranking
       |      |      |                       |
      VQA  Counting Spatial            Counterfactual
       |      |      |                   Distractors
       +------+------+                       |
              |                               |
       Consistency Tests              Top-1 / MRR
              |                               |
              +---------------+---------------+
                              |
                    Statistical Evaluation
                              |
                  Error Taxonomy & Charts
                              |
                    Results and Manifest
```

## 1. Controlled Synthetic Scene Generation

The benchmark programmatically generates images containing:

- One red circle
- One blue square
- Between one and six green triangles
- Controlled horizontal and vertical relationships

Each image has automatically generated ground-truth labels derived from its drawing instructions.

This approach enables precise evaluation without relying on subjective human annotation.

### Scene Difficulty

| Difficulty | Triangle Count |
|---|---|
| Easy | 1–2 |
| Medium | 3–4 |
| Hard | 5–6 |

Difficulty is defined using object density as a proxy, not as a validated measure of human visual difficulty.

## 2. Visual Question Answering

The VQA component evaluates whether BLIP can extract basic visual information from generated scenes.

Questions focus on visual attributes, objects, and image-grounded relationships.

**Evaluation Metric:**

- Normalized strict-match accuracy
- Invalid prediction count
- Task-specific prediction records

The benchmark uses strict normalized matching rather than the official VQA v2 multi-annotator scoring protocol.

## 3. Object Counting

Object counting tests whether the model can identify the exact number of green triangles within an image.

Performance is evaluated across increasing object density.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Exact Count Accuracy | Percentage of predictions matching the true count |
| Mean Absolute Error (MAE) | Average counting error among parseable predictions |
| Invalid Output Rate | Frequency of responses that cannot be interpreted as counts |
| Under-count Rate | Frequency of predicting fewer objects |
| Over-count Rate | Frequency of predicting more objects |

Invalid outputs are counted as incorrect for exact accuracy and excluded from MAE, with valid-response coverage reported separately.

## 4. Spatial Reasoning

Spatial reasoning examines whether the model correctly interprets relationships between objects.

The benchmark evaluates:

- Left versus right
- Above versus below
- Relation-category accuracy
- Spatial confusion patterns

The objective is to determine whether models can distinguish object relationships instead of relying solely on familiar visual layouts.

## 5. Image–Text Alignment and Counterfactual Testing

CLIP evaluates image–text correspondence by ranking candidate captions.

For each synthetic image, the model receives four candidate descriptions:

1. The correct caption
2. A caption with reversed left/right relationships
3. A caption with swapped object colors
4. A caption containing an incorrect triangle count

### Evaluation Metrics

- **Top-1 Accuracy:** Whether the correct caption receives the highest similarity score.
- **Mean Reciprocal Rank (MRR):** How highly the correct caption ranks among alternatives.
- **Distractor Analysis:** Which incorrect caption types are most likely to be selected.

The random-choice reference for four-way caption selection is 25% top-1 accuracy.

A correct ranking demonstrates discrimination within the candidate set, not necessarily general visual reasoning.

## 6. Reasoning Consistency

A particularly important component investigates whether models maintain logically coherent answers when questions are reformulated.

Two forms of consistency are examined:

**Paraphrase Consistency**

Does the model provide equivalent answers when the same question is expressed differently?

**Logical Consistency**

Does the model respond appropriately to logically reversed questions, such as asking whether an object is to the left versus to the right of another object?

The benchmark distinguishes between:

- Answer agreement
- Logical consistency
- Joint correctness

**Key Principle:** A model can produce internally consistent responses while still being factually incorrect. Therefore, consistency and correctness must be evaluated separately.

## 7. Statistical Evaluation and Error Analysis

The notebook incorporates statistical and diagnostic methods to support more reliable interpretation.

### Statistical Analysis

- Image-clustered bootstrap confidence intervals
- Sample-size reporting
- Task-specific performance analysis
- Descriptive majority-label baselines

Bootstrap resampling operates at the image level to account for multiple questions originating from the same image.

### Error Taxonomy

Model errors are categorized into:

- Invalid or failed generations
- Counting errors
- Spatial relation reversals
- Object or attribute mistakes
- Incorrect caption-distractor selections

Visualizations include performance comparisons, count-related analyses, spatial confusion matrices, and a gallery of model failures.

All reported results are generated from actual model predictions rather than manually constructed performance scores.

## 8. Optional Real-World Evaluation: MS COCO

The notebook also includes an optional extension using MS COCO 2017 validation images.

This component evaluates BLIP's counting ability on selected real-world photographs using instance annotations.

Important considerations:

- Crowd annotations are excluded.
- Ground truth represents annotated instances rather than necessarily every visually perceived object.
- Synthetic and real-image results are treated as separate evaluation populations.
- The optional evaluation must be explicitly enabled.

This extension helps investigate how controlled-scene findings may change in more realistic environments.

## 9. Reproducibility

The notebook emphasizes reproducible experimentation through:

- Deterministic synthetic scene generation
- Fixed random seeds and image dimensions
- Greedy answer decoding
- Recorded model identifiers
- Experiment configuration tracking
- Exported predictions and evaluation metrics
- Environment and experiment manifests

### Recommended Environment

Google Colab with a T4 GPU, or another CUDA-compatible environment.

### Core Dependencies

```bash
pip install "transformers>=4.49,<5" \
            "accelerate>=1.3,<2" \
            "safetensors>=0.4" \
            "pycocotools>=2.0.7" \
            "scipy>=1.11" \
            "seaborn>=0.13"
```

The notebook also uses PyTorch, NumPy, Pandas, Matplotlib, and Pillow.

### Run the Benchmark

1. Open the Kaggle notebook or load the notebook into Google Colab.
2. Enable GPU acceleration where available.
3. Install the required dependencies.
4. Run the notebook cells sequentially.
5. Inspect task metrics, charts, and model failure examples.
6. Export the generated results and experiment manifest.

The default evaluation is a small functional pilot, not a comprehensive leaderboard.

## 10. Outputs and Artifacts

The benchmark exports a ZIP archive containing experiment artifacts, including:

- Model predictions
- Task-level evaluation metrics
- Statistical summaries
- Generated charts
- Synthetic scene images
- Reproducibility manifest

These outputs can support independent analysis, experiment replication, and future benchmark extensions.

## Limitations

This benchmark is an exploratory framework rather than a definitive measure of visual intelligence.

Its primary limitations include:

- Synthetic geometric scenes do not represent the full complexity of natural images.
- Fixed layouts may allow models to exploit positional shortcuts.
- Increasing object density does not capture every dimension of visual difficulty.
- Strict answer matching may underestimate semantically correct open-ended predictions.
- BLIP and CLIP cannot be compared as though they solve identical tasks.
- Small pilot samples limit the generalizability of observed performance.
- Public image benchmarks may overlap with model training distributions.

Results should therefore be interpreted as task-specific diagnostic evidence rather than proof that a model possesses or lacks general visual understanding.

## Future Research Directions

Several extensions could strengthen the benchmark:

**1. Complex Visual Scenes**

Introduce overlapping objects, occlusions, cluttered backgrounds, irregular positioning, and unpredictable layouts.

**2. Larger Model Comparisons**

Evaluate additional VLM families under matched tasks and consistent evaluation protocols.

**3. Adversarial Spatial Reasoning**

Test object rearrangements, attribute swaps, and controlled visual perturbations to identify shortcut-based predictions.

**4. Real-World Benchmark Integration**

Extend the evaluation to appropriately prepared VQA v2, GQA, and additional natural-image datasets using their official annotation and scoring conventions.

**5. Robustness and Calibration**

Investigate uncertainty, prediction stability, out-of-distribution generalization, and the relationship between confidence and correctness.

**6. Explainable Multimodal Reasoning**

Explore whether model explanations faithfully reflect the visual evidence supporting their predictions.

## Key Takeaway

**Visual understanding is multidimensional, and no single accuracy score can adequately characterize it.**

A model may perform well at recognizing visual attributes while struggling with counting, relational reasoning, or logically consistent responses.

This benchmark provides a reproducible starting point for investigating those differences and evaluating how reliably vision-language models connect language to visual evidence.

---

## Explore the Notebook

**[Kaggle Notebook](https://www.kaggle.com/code/utkarshjain76/do-vision-llms-actually-understand-images)**

If you find this benchmark useful, consider upvoting the notebook on Kaggle and sharing feedback or ideas for future experiments.

---

*Exploring not just what AI models predict, but how reliably their predictions reflect the visual world.*
