# Latest checkpoint experiment sanity check


## Best fixed-k / best long-interval by framework and model


| framework      | experiment_name                            | model_n   |   freq |   throughput_steps_per_s |   overhead_frac |   p99_step_s |   ckpt_count |
|:---------------|:-------------------------------------------|:----------|-------:|-------------------------:|----------------:|-------------:|-------------:|
| ds-sync        | bloom_1.1b_sync_k500                       | 1b1       |    500 |                 2.34463  |      0.0393901  |     1.24278  |            2 |
| ours           | bloom_1b1_t4_freq_500                      | 1b1       |    500 |                 1.88708  |      0.00488537 |     1.2906   |            2 |
| datastates-llm | bloom_1b1_freq_250                         | 1b1       |    250 |                 1.87233  |      0.00909066 |     1.25229  |            4 |
| ds-async       | bloom_ds_async_like_v4_bloom-1b1_freq_500  | 1b1       |    500 |                 1.73406  |      0.0141574  |     1.34008  |            2 |
| ours           | bloom_3b_t4_freq_1000                      | 3b        |   1000 |                 1.04719  |      0.00407512 |     3.20036  |            1 |
| datastates-llm | bloom_3b_freq_250                          | 3b        |    250 |                 1.03811  |      0.0170094  |     3.08297  |            4 |
| ds-sync        | bloom_3b_sync_k500                         | 3b        |    500 |                 1.01971  |      0.107904   |     3.11028  |            2 |
| ds-async       | bloom_ds_async_like_v4_bloom-3b_freq_500   | 3b        |    500 |                 0.945885 |      0.0140031  |     3.33902  |            2 |
| ds-sync        | bloom_0.56b_sync_k500                      | 560m      |    500 |                 3.8613   |      0.0300239  |     0.736281 |            2 |
| datastates-llm | bloom_560m_freq_1000                       | 560m      |   1000 |                 2.93937  |      0.00240233 |     0.772715 |            1 |
| ours           | bloom_560m_t4_freq_500                     | 560m      |    500 |                 2.92777  |      0.00463364 |     0.746792 |            2 |
| ds-async       | bloom_ds_async_like_v4_bloom-560m_freq_500 | 560m      |    500 |                 2.78105  |      0.0134129  |     0.77254  |            2 |


## Auto vs best fixed


| framework      | experiment_name          | model_n   |   throughput_steps_per_s |   best_fixed_throughput |   auto_over_bestfixed |   overhead_frac |   p99_step_s |
|:---------------|:-------------------------|:----------|-------------------------:|------------------------:|----------------------:|----------------:|-------------:|
| datastates-llm | bloom_1b1_freq_auto      | 1b1       |                 1.35506  |                1.87233  |              0.723729 |       0.252963  |      9.43725 |
| ds-async       | bloom-1b1_async_like_v4  | 1b1       |                 1.63103  |                1.73406  |              0.940586 |       0.0685167 |      2.83593 |
| ours           | bloom_1b1_t4_freq_auto   | 1b1       |                 1.33806  |                1.88708  |              0.709065 |       0.271315  |      9.44489 |
| datastates-llm | bloom_3b_freq_auto       | 3b        |                 0.923788 |                1.03811  |              0.889873 |       0.0977171 |      5.19034 |
| ds-async       | bloom_3b_async_like_v4   | 3b        |                 0.848968 |                0.945885 |              0.897538 |       0.11719   |      5.33835 |
| ours           | bloom_3b_t4_freq_auto    | 3b        |                 0.949052 |                1.04719  |              0.906286 |       0.0980726 |      4.71097 |
| datastates-llm | bloom_560m_freq_auto     | 560m      |                 2.08698  |                2.93937  |              0.710007 |       0.270283  |      6.23999 |
| ds-async       | bloom-560m_async_like_v4 | 560m      |                 2.61268  |                2.78105  |              0.93946  |       0.0642819 |      1.2771  |
| ours           | bloom_560m_t4_freq_auto  | 560m      |                 1.88838  |                2.92777  |              0.644988 |       0.314086  |      7.12841 |


## Controller quality (auto only)


| framework      | experiment_name          | model_n   |   eval_count |   observed_overhead_pct_mean |   observed_overhead_pct_p50 |   tracking_error_pct_mean |   within_band_rate |   num_freq_changes |   avg_chosen_freq_all |   final_freq |
|:---------------|:-------------------------|:----------|-------------:|-----------------------------:|----------------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|-------------:|
| datastates-llm | bloom_1b1_freq_auto      | 1b1       |           25 |                      4.41601 |                     4.06397 |                   1.23564 |           0.24     |                 19 |               41.12   |           55 |
| ds-async       | bloom-1b1_async_like_v4  | 1b1       |           10 |                      6.50968 |                     5.11506 |                   1.8425  |           0.4      |                  6 |              103      |          116 |
| ours           | bloom_1b1_t4_freq_auto   | 1b1       |           26 |                     10.2896  |                     3.96468 |                   7.38286 |           0.230769 |                 20 |               39.2692 |           39 |
| datastates-llm | bloom_3b_freq_auto       | 3b        |           11 |                      6.1894  |                     5.20245 |                   2.11171 |           0.181818 |                  9 |              102.273  |          144 |
| ds-async       | bloom_3b_async_like_v4   | 3b        |            9 |                      7.78018 |                     6.02064 |                   3.09686 |           0.222222 |                  7 |              119.556  |          135 |
| ours           | bloom_3b_t4_freq_auto    | 3b        |           13 |                      5.11825 |                     4.37285 |                   1.11575 |           0.615385 |                  5 |               82.8462 |           93 |
| datastates-llm | bloom_560m_freq_auto     | 560m      |           31 |                      4.01383 |                     3.73343 |                   1.47534 |           0.225806 |                 24 |               31.5806 |           24 |
| ds-async       | bloom-560m_async_like_v4 | 560m      |            9 |                      6.86131 |                     5.09834 |                   2.01263 |           0.333333 |                  6 |              109.222  |          125 |
| ours           | bloom_560m_t4_freq_auto  | 560m      |           33 |                      4.27305 |                     4.18027 |                   1.04145 |           0.454545 |                 18 |               30.2121 |           36 |
