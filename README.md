# Adolescent Digital Habit Analysis

Does *how much* time adolescents spend on screens matter, or *what* they do on them?

This project re-analyses the raw survey data from Pérez-Chada et al. (2023, *PLOS ONE*) to test whether **total screen time** or **device type** better explains adolescent **sleep debt** and **classroom attention problems**. The original study grouped device types together. This analysis looks at each one separately, in particular social media.

## Research questions

| | Question | Alternative hypothesis |
|---|---|---|
| **RQ1** | Is total daily screen time associated with weekend sleep debt? | Higher total screen time is associated with greater sleep debt. |
| **RQ2** | Is device type associated with classroom attention problems? | Passive social media use is associated with worse attention; active console gaming is not. |

## Data

- **Source:** Pérez-Chada, D. et al. (2023). *Screen use, sleep duration, daytime somnolence, and academic failure in school-aged adolescents.* PLOS ONE 18(2): e0281379. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC9928097/)
- **Size:** 3,624 adolescents in the raw dataset. This analysis uses the full raw dataset rather than the stricter subset (N = 1,257) used in the original paper. Only duplicate and incomplete rows are removed.
- **Variables:** self-reported daily hours on video games, tablet/phone games, social media and TV/online video; weekday and weekend sleep duration; an ordinal attention-problems score (1 = best, 4 = worst).

The dataset is not included in this repository. Place it at `data/repositorio_translated.csv` (column names translated to English).

## Methods

The screen-time variables are heavily right-skewed, and the attention score is ordinal, so **non-parametric methods** are used throughout.

1. **Device type vs attention:** Spearman rank correlation for each device type, with R² reported to measure practical effect size alongside the p-value.
2. **Total screen time vs sleep debt:** Spearman correlation, where sleep debt = weekend sleep − weekday sleep.
3. **Behavioural profiles:**
   - Device-time features are standardised, then clustered with **K-means**. The number of clusters is assessed with **silhouette scores** for k = 2, 3, 4.
   - Clusters are named from their centroids (highest social-media time, highest gaming time, balanced), not from arbitrary cluster IDs.
   - A **Kruskal-Wallis H-test** checks whether attention differs across profiles, and **Dunn's post-hoc test with Bonferroni correction** identifies which profiles differ.

## Key results

- **RQ1:** Total screen time is statistically significant but **practically negligible** for sleep debt (r = 0.087, R² = 0.0075, p < 0.001). It explains less than 1% of the variance, so the "displacement hypothesis" gets no practical support.
- **RQ2:** All device types correlate significantly with attention problems, but only **social media** has a meaningful effect (r = 0.178, R² = 0.032). That is more than double any other device type, all of which have R² < 0.01.
- **Profiles:** attention differs significantly across the three profiles (Kruskal-Wallis H = 25.3, p < 0.001). Dunn's test shows that **only heavy social media users** have significantly worse attention, compared with both balanced users (p = 0.019) and heavy gamers (p < 0.001). Heavy gamers do not differ from balanced users (p = 1.000).

**Implication:** the results support targeting passive social media use rather than blanket screen-time limits.

## Limitations

- **Correlational, not causal:** the data are cross-sectional, so the results show associations, not effects.
- **Self-reported** screen time and sleep, which may introduce measurement error.
- **k = 3 was chosen over k = 2** despite a slightly higher silhouette score for k = 2 (0.646 vs 0.610), so that active gamers and passive social media users could be separated.
- K-means assumes roughly spherical clusters, which zero-inflated variables (most students play no console games) only partly satisfy.
- **Implausible screen-time values:** some students report a total daily screen time above 24 hours, up to about 120 hours. This most likely reflects self-reporting errors or overlapping use of several devices at once (for example, TV on while scrolling) being counted more than once. These rows were kept to stay consistent with the raw dataset, but they may inflate the spread of the screen-time variables.
- **Out-of-range attention scores:** the attention scale runs from 1 (best) to 4 (worst), but a few records have a score of 0, which may be a code for missing data. These records were not removed in this analysis.
- A natural next step is to re-run the analysis after excluding these records, to check that the conclusions hold.

## Running the analysis

```bash
pip install -r requirements.txt
python analysis.py --data data/repositorio_translated.csv --out figures
```

The statistics are printed to the console, and four figures are saved to `figures/`.
