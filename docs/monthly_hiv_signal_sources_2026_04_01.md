# Monthly HIV Signal Sources For Phase 3

This note records the strongest public monthly HIV signal sources I found that are relevant to the Philippine Phase 3 scaffold. These are not all equal in quality. The order below reflects scientific utility for the current model.

## 1. Monthly HARP registry bulletins

- Source family: DOH Epidemiology Bureau / PNAC monthly HIV/AIDS & ART Registry of the Philippines bulletins
- Example: [HARP June 2023](https://pnac.doh.gov.ph/wp-content/uploads/2023/10/HARP-June-2023.pdf)
- Why this matters:
  - monthly diagnosed counts
  - monthly ART enrollment and treatment information
  - monthly deaths
  - national totals with regional breakouts
- Use in the model:
  - strongest public monthly national HIV signal
  - should be preferred over annual interpolation whenever the bulletin month is explicit

## 2. STI/HIV Denominator Surveillance System monthly laboratory reporting

- Source family: DOH Epidemiology Bureau STI/HIV Denominator Surveillance System
- Program brief: [DOH Program Briefer 2020 Updated Final](https://caro.doh.gov.ph/wp-content/uploads/2021/11/Program-Briefer_2020_UPDATED_FINAL.pdf)
- Why this matters:
  - monthly number of HIV tests performed
  - positivity and referral signal
  - laboratory and blood-bank reporting channel
- Scientific importance:
  - the current Philippine bottleneck is testing and diagnosis
  - this source family is therefore more relevant to `U -> D` than many literature-derived determinants
- Use in the model:
  - monthly testing denominator anchor
  - monthly testing-pressure covariate
  - shock-sensitive testing-disruption measurement during COVID windows

## 3. Regional STI/HIV monthly surveillance reports

- Source family: DOH Regional Epidemiology and Surveillance Unit portals
- Example hub: [CRE@TE onLINE reports](https://www.resu.online/reports)
- Why this matters:
  - explicit regional monthly STI/HIV surveillance reporting
  - useful when national monthly anchors are sparse and Phase 3 needs region-level timing
- Use in the model:
  - regional monthly anchor layer
  - mixed-frequency observation support for region-year and region-month constraints

## Practical conclusion

The current model is still dominated by annual anchors because true monthly HIV support is scarce. The most promising path is not more generic literature harvesting. It is ingestion of:

1. monthly HARP bulletins,
2. monthly denominator surveillance reports,
3. regional monthly STI/HIV surveillance reports.

These sources are directly relevant to the identified failure mode: diagnosis and treatment-entry timing are still being modeled worse than simple compartmental and carry-forward baselines.
