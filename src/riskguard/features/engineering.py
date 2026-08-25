"""
riskguard.features.engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feature engineering pipeline (Phase 2+).

Transforms applied here:
* Scale ``Amount`` and ``Time`` (V-features are already PCA-standardised).
* Optionally add ``HourOfDay`` from the ``Time`` column.

All transformers are sklearn-compatible so they can be embedded in a Pipeline.
"""

# TODO (Phase 2): Implement StandardScaler for Amount / Time.
# TODO (Phase 2): Add hour-of-day feature derived from Time.
