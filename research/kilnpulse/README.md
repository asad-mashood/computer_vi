# KilnPulse: Monitoring Whether Individual Brick Kilns Are Running

**Authors:** Muhammad Ismail, Adeel Asghar, Asad Masood
**Affiliation:** Computer Science Department, COMSATS University Islamabad, Wah, Pakistan

Brick kilns are a major source of air pollution in Pakistan. Maps show where kilns are, but not which ones are running. KilnPulse estimates, for each mapped kiln and in every 12-day window, whether it is running, using only free data in Google Earth Engine.

## Approach

- **Sentinel-1 (radar):** change in the kiln yard, which works through smog and cloud.
- **Sentinel-2 (optical):** vegetation, brick-red and dark-surface contrasts.
- **Landsat 8/9 (thermal):** kiln warmth.
- **Random Forest** combines the clues, and a **two-state hidden Markov model** smooths each kiln's sequence.

Every clue compares a kiln with its own surrounding fields.

## Files

- [`KilnPulse_abstract_intro.pdf`](KilnPulse_abstract_intro.pdf): abstract and introduction.

## Status

Work in progress. The abstract still has bracketed placeholders for some values (`[1]` to `[4]`), so the figures are not final.

**Index terms:** brick kilns, activity monitoring, Sentinel-1, Sentinel-2, Landsat, air pollution, hidden Markov model
