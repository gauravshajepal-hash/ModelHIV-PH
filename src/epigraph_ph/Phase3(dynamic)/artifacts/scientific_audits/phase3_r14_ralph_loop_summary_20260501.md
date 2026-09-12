# Phase 3 R14 Ralph Loop Summary - 2026-05-01

This audit compares the R14 two-factor monthly program state, horizon selector, bounded long-horizon residual calibration, and train-supported flow-growth-bound variants under the same R13 priority queue.

| branch | R13-006 family | R13-006 mean | R13-006 R10 delta | R13-006 h3 delta | R13-006 h5 delta | R13-050 mean | R13-050 R10 delta | R13-050 h1 delta | R13-050 h3 delta | R13-050 h5 delta | decision | blockers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| R14 raw | r14_two_factor_program_process | 0.175737 | 0.053201 | 0.000891 | 0.105511 | 0.368021 | 0.042229 | -0.013827 | 0.037272 | 0.103242 | keep_as_diagnostic | matched_r10_gate_failed |
| R14 selector | r14_two_factor_horizon_selector_process | 0.169290 | 0.046754 | -0.016426 | 0.109933 | 0.361687 | 0.041347 | -0.013827 | 0.034137 | 0.103731 | keep_as_diagnostic | matched_r10_gate_failed |
| R14B unbounded | r14_program_long_horizon_calibrated_process | 0.173015 | 0.050479 | 0.002050 | 0.098908 | 0.361841 | 0.040029 | -0.013845 | 0.036472 | 0.097459 | keep_as_diagnostic | matched_r10_gate_failed |
| R14B bounded | r14_program_long_horizon_calibrated_process | 0.166857 | 0.044321 | 0.005805 | 0.082837 | 0.346853 | 0.034809 | -0.014330 | 0.034321 | 0.084437 | keep_as_diagnostic | matched_r10_gate_failed |
| R14B flow/growth bounded | r14_program_long_horizon_calibrated_process | 0.166857 | 0.044321 | 0.005805 | 0.082837 | 0.346853 | 0.034809 | -0.014330 | 0.034321 | 0.084437 | keep_as_diagnostic | matched_r10_gate_failed |

Interpretation: bounded R14B is the best branch in this pass. It improves the R13-006 program 3/5-year R10 delta from 0.053201 to 0.044321 and the full sentinel R13-050 R10 delta from 0.042229 to 0.034809. It still fails matched R10, mainly at h5. The diagnosis-flow channel fails closed, and relaxing the flow retention upper bound to train-supported observed growth does not change the queue, so the next scientific step is not more residual calibration; it is a structural diagnosis-flow/incidence transition module.

Current promoted count remains 11/50; the publication sentinel remains diagnostic, not a champion.
