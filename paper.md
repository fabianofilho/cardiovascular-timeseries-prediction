# Series Length, Not Model Choice, Drives Forecasting Accuracy of Cardiovascular Mortality in São Paulo, Brazil: A Benchmark of Foundation, Classical, and Boosting Models with Measured Uncertainty

**Fabiano B. N. Filho¹ · Isabela Venancio da Silva¹ · Alexandre Chiavegatto Dias Porto Filho¹**

¹ Faculdade de Saúde Pública, Universidade de São Paulo (FSP-USP)

---

## Abstract

Cardiovascular diseases (CVDs) remain the leading cause of death in Brazil, accounting for approximately 30% of all deaths. Accurate forecasting of cardiovascular mortality supports public health planning and resource allocation. Foundation models pre-trained on large time series corpora have shown strong zero-shot performance across forecasting benchmarks, and recent applied studies frequently report them as outperforming classical alternatives. Such rankings, however, are typically reported as point estimates, without any measure of how much of the observed gap is sampling noise.

We benchmarked five forecasting approaches, SARIMA, Prophet, TimesFM, XGBoost, and CatBoost, on monthly cardiovascular mortality in the state of São Paulo, Brazil, using 168 months of real records from the Brazilian Mortality Information System (SIM/DataSUS, January 2010 to December 2023; 1,217,427 cardiovascular deaths). Rolling origin cross-validation with a 6-month horizon and a 60-month minimum training window produced 103 windows and 618 out-of-sample predictions per model. Every comparison carries uncertainty, estimated by a block bootstrap that resamples entire windows (B = 10,000) and by the Diebold-Mariano test with the Harvey correction applied per horizon.

Prophet (sMAPE 4.70%, 95% CI 4.17 to 5.27), SARIMA (4.80%, 4.29 to 5.33), and TimesFM (4.83%, 4.32 to 5.39) were statistically indistinguishable: the spread between them (0.13 percentage points, pp) was an order of magnitude smaller than the mean confidence interval width (1.07 pp), all three paired differences contained zero (p = 0.17 to 0.77), and none of the 18 Diebold-Mariano cells reached significance. Boosting models were detectably worse (38 of 42 cells at p < 0.05). Holding test dates fixed, nine additional years of training data reduced sMAPE by 1.23 to 1.71 pp, an effect roughly ten times larger than any between-model difference. A sliding-window experiment isolated the one regime where the foundation model has an advantage: when training history is truncated to 60 months, SARIMA and Prophet lose 0.66 and 0.69 pp while TimesFM is unaffected (p = 0.42), and TimesFM leads.

We conclude that model selection among these three families is an operational decision rather than a statistical one, that data extension delivered far more than model substitution, and that the defensible role of a zero-shot foundation model in this setting is robustness to short history rather than superior accuracy.

**Keywords:** cardiovascular mortality, time series forecasting, foundation models, TimesFM, SARIMA, Prophet, forecast uncertainty, Diebold-Mariano test, DataSUS, epidemiological surveillance, Brazil

---

## 1. Introduction

Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible for approximately 17.9 million deaths per year [1]. In Brazil, CVDs have been the primary cause of mortality since the 1960s, accounting for roughly 30% of all deaths in recent decades, surpassed only by COVID-19 in 2020 and 2021 [2, 3]. The state of São Paulo, the most populous in Brazil with over 46 million inhabitants, bears a substantial share of this burden [4].

Despite a sustained decline in age-standardized CVD mortality rates since the 1990s, attributed to improved risk factor control, expanded primary care coverage, and advances in acute treatment [5], the absolute number of cardiovascular deaths continues to rise due to population growth and aging [6]. In our own series, annual cardiovascular deaths in São Paulo rose from 79,933 in 2010 to 95,538 in 2023. This underscores the ongoing need for forecasting tools that can anticipate mortality trends and inform resource allocation within Brazil's universal public health system (SUS).

Time series forecasting has a long history in epidemiological surveillance. Classical statistical models such as Seasonal ARIMA (SARIMA) have been widely used for modeling disease incidence and mortality [7], and Meta's Prophet has gained adoption for its interpretable decomposition of trend and seasonality [8]. The emergence of foundation models represents a different proposition: pre-trained on massive and diverse time series corpora, they produce competitive zero-shot forecasts on unseen datasets without task-specific training [9]. TimesFM, a decoder-only transformer with 200 million parameters pre-trained on approximately 100 billion real-world time points, has demonstrated strong zero-shot performance across multiple benchmarks [9].

A methodological gap accompanies this promise. Applied comparisons in health forecasting commonly report a ranking of models by a point estimate of aggregate error, and interpret the ordering as a finding. Yet backtesting predictions are not independent observations: they come from overlapping rolling windows, and within each window the horizons share an origin. Treating them as independent understates uncertainty, and the resulting ranking can be entirely a product of sampling noise. Whether an observed gap of a few tenths of a percentage point constitutes evidence is an empirical question that requires an interval, not a point.

This study addresses both the applied and the methodological question. We compare five forecasting approaches on a monthly cardiovascular mortality series derived from real SIM/DataSUS records for São Paulo (2010-2023), and we attach a confidence interval to every comparison using a block bootstrap over whole windows and the Diebold-Mariano test. We further decompose where forecasting accuracy actually comes from, by holding test dates fixed while varying the amount of training history, and by testing monthly temperature as an exogenous variable under an explicitly leakage-free design.

This work extends an earlier round of the same benchmark restricted to 60 months (2019-2023), in which TimesFM ranked first. That ranking did not survive the addition of uncertainty and of nine further years of data, and the present paper reports why.

---

## 2. Methods

### 2.1 Data Source and Preprocessing

This study used individual death records from the Brazilian Mortality Information System (Sistema de Informações sobre Mortalidade, SIM), maintained by the Ministry of Health and accessible through the DataSUS platform [10]. Records were extracted for the state of São Paulo (UF = SP) covering January 2010 to December 2023. Deaths were filtered by underlying cause using ICD-10 codes I00-I99 (diseases of the circulatory system) [11].

From a total of 4,317,224 raw mortality records, 1,217,427 were classified as cardiovascular deaths and aggregated into a monthly series of 168 points. The series has a mean of 7,247 deaths per month (median 7,177), a minimum of 5,811 (February 2011), and a maximum of 9,582 (January 2022).

Extraction was performed by `scripts/extract_sim_real.py` and audited by `scripts/validate_real_dataset.py`, which verifies that the ICD column exists, that every retained record matches the requested prefix, that at least three distinct codes are present, that the aggregated series is non-degenerate, and that the extraction metadata is not flagged as synthetic. All nine checks passed (`results/data_reality_report_2010_2023.json`). The aggregated series is versioned in the repository; individual records are not.

São Paulo has one of the highest death registration completeness rates in Brazil, above 95%, which supports the reliability of the series. The series uses raw SIM data without garbage code redistribution, which should be considered when comparing against Global Burden of Disease estimates.

### 2.2 Forecasting Models

Five approaches were evaluated:

- **SARIMA:** Seasonal Autoregressive Integrated Moving Average with order (1,1,1) and seasonal order (0,1,1,12), implementing yearly seasonality through seasonal differencing [7].
- **Prophet:** additive decomposition model, configured with yearly seasonality enabled and weekly and daily seasonalities disabled, appropriate for monthly data [8].
- **TimesFM:** decoder-only foundation model [9], using the public `google/timesfm-2.5-200m-pytorch` checkpoint with a 512-point context window, applied zero-shot without fine-tuning on the target series.
- **XGBoost** [16] and **CatBoost** [17]: gradient boosting regressors wrapped in a recursive multi-step forecaster (skforecast `ForecasterRecursive`) with lags 1 to 12.

An earlier version of this pipeline clipped SARIMA and boosting forecasts to a range derived from the training window, a safeguard that Prophet and TimesFM never received. Because this asymmetry contaminates any comparison between families, the clip was removed and replaced by a diagnostic warning. On this series the change is inert: none of the 9,888 forecasts produced across the four benchmark runs reported here fell outside the former clipping bounds.

### 2.3 Backtesting Strategy

Model performance was evaluated by rolling origin cross-validation, which simulates operational forecasting by iteratively advancing the training window and generating out-of-sample predictions.

| Parameter | Value |
|---|---|
| Forecast horizon | 6 months |
| Minimum training window | 60 months |
| Window type | Expanding (60 to 162 months of training) |
| Step | 1 month |
| Total rolling windows | 103 |
| Total predictions per model | 618 (103 × 6) |

The 60-month minimum provides at least four effective seasonal cycles after the double differencing implied by the SARIMA specification. Every model was evaluated on identical windows and identical test dates, which is what makes the paired analysis in Section 2.5 valid.

### 2.4 Evaluation Metrics

Predictive accuracy was assessed with mean absolute error (MAE), root mean squared error (RMSE), and symmetric mean absolute percentage error (sMAPE). MAE is reported in deaths per month and is directly interpretable; RMSE penalizes larger deviations; sMAPE is scale-independent and therefore comparable across series and periods.

### 2.5 Uncertainty Quantification

The 618 predictions per model are not independent. They arise from 103 overlapping rolling windows, and the six horizons within a window share a training origin. Resampling individual predictions would therefore overstate precision. Two complementary procedures were used instead.

**Block bootstrap over windows.** Whole windows, with their six horizons kept together, were resampled with replacement, B = 10,000 replicates, seed 20260817. In each replicate the same resampled windows were applied to every model, preserving pairing. Confidence intervals are the 2.5th and 97.5th percentiles of the replicate distribution. Paired differences between models were computed within replicate, yielding an interval for the difference itself.

**Diebold-Mariano test.** The equality of predictive accuracy was tested per horizon using the Diebold-Mariano statistic [18] with the small-sample correction of Harvey, Leybourne and Newbold [19], with absolute error as the loss function.

**Pre-declared decision criterion.** To avoid selecting a favorable statistic after the fact, an improvement was declared confirmed only if the 95% interval of the paired difference excluded zero **and** the Diebold-Mariano test reached p < 0.05 in at least 3 of the 6 horizons. Results that met the first condition but not the second are reported as suggestive and explicitly not confirmed.

### 2.6 Series Length Experiment

Aggregate error is not comparable across rounds that use different test periods, because periods differ in intrinsic difficulty. Two designs were used to isolate the effect of training history.

First, the subset of predictions falling on the test dates shared with the earlier 60-month round (January 2021 to December 2023) was extracted and compared directly, holding the target dates fixed while the amount of training history differs.

Second, the full benchmark was rerun on the same 103 origins with training truncated to the most recent 60 months (`--max-train-size 60`), so that the only difference between runs is how much past each model may use.

### 2.7 Exogenous Temperature

Low ambient temperature is the principal candidate driver of cardiovascular seasonality, acting through elevated blood pressure and platelet aggregation [12, 13]. Monthly minimum temperature was obtained from the National Institute of Meteorology (INMET) automatic station A701 (São Paulo, Mirante de Santana), aggregated from hourly records to monthly means of daily minima, with 168 complete months and no gaps. Days with fewer than 18 valid hours and months with fewer than 20 valid days were discarded before aggregation.

Using an observed future covariate would leak information unavailable at forecast time. Three policies were therefore implemented and compared:

- **climatology** (the policy used for inference): the future covariate is the month-of-year mean recomputed for each window from the exogenous series truncated at the end of that window's training set. No value from after the training end is ever visible.
- **lag12**: the observed value twelve months earlier, also drawn from the truncated series.
- **observed**: the true future value. This leaks by construction and is reported only as a labelled ceiling scenario, quantifying how much a perfect weather forecast could add.

Only models that natively accept exogenous regressors received the covariate (SARIMA, XGBoost, CatBoost). Prophet and TimesFM were excluded from the exogenous comparison rather than silently ignoring the input.

---

## 3. Results

### 3.1 Aggregate Performance

**Table 1.** Comparative performance for monthly cardiovascular mortality in São Paulo, 2010-2023. All models evaluated on the same 618 out-of-sample predictions from 103 rolling origin windows with a 6-month horizon. Intervals are 95% block bootstrap over windows, B = 10,000.

| Model | MAE | RMSE | sMAPE (%) | 95% CI of sMAPE | CI width |
|---|---:|---:|---:|:---:|---:|
| Prophet | 355.17 | 515.76 | 4.70 | [4.17, 5.27] | 1.10 pp |
| SARIMA | 363.26 | 508.11 | 4.80 | [4.29, 5.33] | 1.04 pp |
| TimesFM | 366.04 | 513.39 | 4.83 | [4.32, 5.39] | 1.06 pp |
| CatBoost | 500.90 | 673.56 | 6.59 | [5.92, 7.27] | 1.35 pp |
| XGBoost | 526.58 | 704.47 | 6.94 | [6.26, 7.65] | 1.39 pp |

With a mean of roughly 7,250 deaths per month, an MAE of 355 corresponds to a relative error near 4.9%.

### 3.2 The Leading Three Models Are Statistically Indistinguishable

The spread in sMAPE among Prophet, SARIMA, and TimesFM is 0.13 pp, against a mean confidence interval width of 1.07 pp, a ratio of 0.12. All three paired differences contain zero, and no Diebold-Mariano cell reaches significance.

**Table 2.** Paired differences among the three leading models. Bootstrap over identical resampled windows, B = 10,000.

| Comparison | Difference (pp) | 95% CI | p (bootstrap) | DM cells at p < 0.05 |
|---|---:|:---:|---:|:---:|
| TimesFM minus SARIMA | +0.037 | [-0.262, +0.303] | 0.768 | 0 of 6 |
| TimesFM minus Prophet | +0.132 | [-0.056, +0.319] | 0.171 | 0 of 6 |
| SARIMA minus Prophet | +0.095 | [-0.157, +0.367] | 0.512 | 0 of 6 |

In the earlier 60-month round, the point ranking placed TimesFM first (7.30%) and Prophet last (7.68%), with a spread of 0.38 pp against a mean interval width of 2.45 pp. Extending the series inverted that ordering while narrowing the intervals by a factor of 2.3. An ordering that reverses when more data arrives, and whose intervals overlap in both rounds, is not a finding about the models.

### 3.3 Boosting Is Detectably Worse

XGBoost and CatBoost sit 1.8 to 2.2 pp above the leading group. All six comparisons against the leading three yield p_bootstrap = 0.000, and 38 of 42 Diebold-Mariano cells fall below 0.05. Unlike the ordering within the leading group, this difference is real by the pre-declared criterion. Recursive boosting over lags, without exogenous information, is not competitive on this series.

### 3.4 Training History Dominates Model Choice

Restricting attention to the test dates shared with the earlier round isolates the effect of training length.

**Table 3.** Same test dates (January 2021 to December 2023), different training history.

| Model | sMAPE, 24 to 54 months of training | sMAPE, 132 to 168 months | Gain |
|---|---:|---:|---:|
| Prophet | 7.68 | 5.97 | 1.71 pp |
| SARIMA | 7.45 | 6.18 | 1.27 pp |
| TimesFM | 7.30 | 6.07 | 1.23 pp |

Nine additional years of training reduce error by 1.23 to 1.71 pp on identical targets, roughly ten times the 0.13 pp that separates the models here and four times the 0.38 pp that separated them in the earlier round. Within the range of options available to a practitioner, extending the series was worth far more than exchanging the model.

### 3.5 The Foundation Model's Advantage Is Robustness to Short History

Rerunning the benchmark on the same 103 origins with training truncated to 60 months isolates the value of long history for each model.

**Table 4.** Expanding versus sliding 60-month training window, identical test origins.

| Model | Expanding | Sliding 60 | Difference (pp) | 95% CI | Verdict |
|---|---:|---:|---:|:---:|---|
| SARIMA | 4.80 | 5.46 | -0.66 | [-1.008, -0.346] | History helps, confirmed |
| Prophet | 4.70 | 5.39 | -0.69 | [-1.159, -0.219] | History helps, suggestive |
| TimesFM | 4.83 | 4.74 | +0.09 | [-0.125, +0.307] | Indifferent (p = 0.42) |
| XGBoost | 6.94 | 6.87 | +0.07 | [-0.213, +0.354] | Indifferent |
| CatBoost | 6.59 | 6.64 | -0.05 | [-0.288, +0.176] | Indifferent |

Discarding history beyond five years costs SARIMA and Prophet substantially and leaves TimesFM unchanged, which is consistent with its design: a bounded context window with internal normalization, and knowledge acquired in pre-training rather than from the target series. In the truncated regime TimesFM leads at 4.74% against 5.46% for SARIMA, with Diebold-Mariano below 0.05 at horizons 3, 4, and 5.

This is the most informative characterization of the foundation model obtained here. It is not more accurate when long history is available; it is insensitive to the absence of history. The operational scenario it fits is forecasting many indicators or many territories where no single series is long.

### 3.6 Temperature Helps, Below the Confirmation Threshold

**Table 5.** Effect of monthly minimum temperature as an exogenous covariate under the leakage-free climatology policy.

| Model | Without | With temperature | Gain (pp) | 95% CI of difference | DM cells at p < 0.05 |
|---|---:|---:|---:|:---:|:---:|
| SARIMA | 4.80 | 4.66 | 0.136 | [0.034, 0.257] | 1 of 6 |
| CatBoost | 6.59 | 6.33 | 0.258 | [0.073, 0.452] | 1 of 6 |
| XGBoost | 6.94 | 6.55 | 0.382 | [0.222, 0.543] | 2 of 6 |

All three intervals exclude zero, but none reaches the pre-declared threshold of significance in at least three horizons. Under the criterion fixed before the analysis, the effect is suggestive and not confirmed.

The labelled ceiling scenario is more informative than the point gains. Replacing the climatological future covariate with the true observed future temperature, which no operational forecast could access, improves SARIMA by only a further 0.03 pp (4.66 to 4.63). The predictive content of monthly temperature in this series is therefore almost entirely its climatology, information that a seasonal model already carries. This bounds what any investment in meteorological forecasting could contribute at monthly resolution, and suggests that finer temporal resolution, where cold waves are resolved rather than averaged away, is where the covariate could still matter.

### 3.7 Seasonality

The series exhibits a stable seasonal pattern across all fourteen years, peaking in July (mean 8,247 deaths) and reaching its trough in February (mean 6,273), an amplitude of 27.2% around the mean. The pattern is consistent with established evidence on cold-related cardiovascular risk [12, 13]. The COVID-19 period introduced level anomalies without disrupting the seasonal shape.

### 3.8 Figures

- **Figure 1** (`images/fig1_time_series_2010_2023.png`): monthly cardiovascular mortality, January 2010 to December 2023.
- **Figure 2** (`images/fig2_forecast_comparison_2010_2023.png`): observed versus predicted values across the backtesting period.
- **Figure 3** (`images/fig3_smape_ci_2010_2023.png`): sMAPE by model with 95% bootstrap intervals.
- **Figure 4** (`images/fig4_smape_by_horizon_2010_2023.png`): error decomposition by forecast horizon.
- **Figure 5** (`images/fig7_seasonal_profile_2010_2023.png`): seasonal profile by month of year.
- **Figure 6** (`images/fig8_exog_effect_2010_2023.png`): effect of the temperature covariate by model.

All figures are generated by `scripts/generate_figures_2010_2023.py` from observed data only. Figures from the earlier round contained 24 months of synthesized values for 2019 and 2020 and have been superseded.

---

## 4. Discussion

This study compared a time series foundation model against classical statistical and gradient boosting approaches for forecasting cardiovascular mortality from Brazilian administrative data, and attached an uncertainty interval to every comparison. The central result is negative in a useful way: among Prophet, SARIMA, and TimesFM there is no detectable difference, and the ordering reported in our own earlier round on a shorter series did not survive either the addition of data or the addition of intervals.

### 4.1 Rankings Without Intervals Are Not Findings

The earlier round of this benchmark reported TimesFM first by 0.38 pp of sMAPE. That gap was smaller than the confidence interval of any single model by a factor of six, and the ordering reversed when the series was extended. Had the first round been published as a ranking, it would have made a claim that the same pipeline contradicted three weeks later using more data.

This is not specific to our series. Backtesting predictions from overlapping rolling windows are strongly dependent, and aggregate error computed over a few hundred such predictions carries an interval on the order of a percentage point, while published gaps between modern forecasting methods are frequently a few tenths. Any comparison in this regime that reports only point estimates is underpowered by construction, whether or not the authors are aware of it. We would encourage applied forecasting studies in health to report the width of the interval alongside the gap between models, and to treat a gap smaller than the interval as what it is.

### 4.2 Where the Accuracy Actually Came From

Holding test dates fixed, extending the training series from at most 54 months to at most 168 months reduced error by 1.23 to 1.71 pp, roughly ten times the spread among models. For a surveillance team, this reorders the priorities: the effort of retrieving and validating a longer historical extraction dominates the effort of implementing a more sophisticated model, and the two are not close.

This also reframes what a benchmark of this kind should report. Aggregate error across rounds with different test periods is not comparable, and our own aggregate figures (7.3% then, 4.8% now) overstate the improvement because the test period changed with the training period. The paired comparison on shared dates is the only valid quantity, and it is smaller than the naive difference.

### 4.3 The Operational Role of Foundation Models

The sliding window experiment gives the foundation model a defensible role that the aggregate ranking obscured. TimesFM is indifferent to the amount of local history, which follows from its architecture: a bounded context with internal normalization, and knowledge acquired in pre-training. SARIMA and Prophet, which estimate their parameters from the target series alone, degrade measurably when history is truncated.

The practical consequence is that the choice depends on the deployment regime, not on a global ranking. For a single well-established indicator with a long series, the classical models match the foundation model and cost far less to run, and the choice should be made on computational cost, transparency, and maintainability. For a surveillance system that must forecast many indicators, municipalities, or newly defined strata, where no individual series is long, zero-shot robustness is worth more than it appears in an aggregate table, and eliminates per-series model selection and hyperparameter tuning.

### 4.4 Temperature and the Limits of Monthly Resolution

Temperature improved every model that received it, with intervals excluding zero, yet failed the pre-declared horizon-level criterion. Rather than report the favorable half of that result, we note that the ceiling scenario explains why the effect is small: a perfect future temperature adds only 0.03 pp over its climatology. At monthly aggregation, temperature carries little information beyond the seasonal shape a model already captures. Cold waves, which are the plausible mechanism for acute cardiovascular events, occur on a scale of days and are averaged out. Weekly or daily aggregation is therefore the condition under which this covariate could become informative, not a better source of monthly temperature.

### 4.5 Public Health Implications

The seasonal amplitude of 27.2% between the July peak and the February trough is substantial for a subtropical setting with mild winters, and stable enough across fourteen years to support anticipatory planning: seasonal allocation of coronary and stroke unit capacity, emergency service staffing, and medication supply. The forecast error achieved here, roughly 4.7 to 4.8% sMAPE at horizons up to six months, is compatible with planning at state level, though not with fine-grained operational decisions.

### 4.6 Limitations

- **Counts rather than rates.** The series models absolute deaths, so part of the trend absorbed by long history is demographic growth and aging rather than epidemiological change. Modeling rates per resident population would separate the two and is a priority for the next iteration.
- **Single geographic unit.** State-level aggregation masks heterogeneity across municipalities and regional health departments.
- **No stratification.** Age group, sex, and ICD-10 subcategory (ischemic heart disease, cerebrovascular disease, heart failure) have distinct seasonal drivers that the aggregate obscures.
- **Zero-shot only.** TimesFM was not fine-tuned. The extended series now satisfies the data prerequisite for fine-tuning, which remains pending on GPU availability.
- **Pandemic period.** The series includes COVID-19, which introduced level shifts and possible cause-of-death misclassification.
- **Monthly resolution.** As discussed, this bounds what exogenous meteorological information can contribute.
- **Raw SIM data.** No garbage code redistribution was applied, which affects comparability with Global Burden of Disease estimates.
- **Data latency.** SIM records take months to consolidate, which constrains real-time application.

### 4.7 Future Directions

The ordering of the following reflects the finding that data, not architecture, moved the error most.

1. **Rates instead of counts**, using resident population denominators.
2. **Stratification** by age group (0-39, 40-59, 60-79, 80+), sex, and ICD-10 subcategory, with an ensemble weighted by stratum share.
3. **Weekly aggregation**, which is the regime where the temperature covariate could plausibly clear the confirmation threshold.
4. **Fine-tuning TimesFM** on the extended series, now that the length prerequisite is met.
5. **Finer spatial granularity**, by municipality or regional health department, which is also the regime where the short-history robustness of the foundation model would be tested at scale.
6. **Operational integration** into existing surveillance platforms.

---

## 5. Conclusion

We benchmarked five forecasting approaches for monthly cardiovascular mortality in São Paulo using fourteen years of real records from Brazil's Mortality Information System, with a confidence interval attached to every comparison. Prophet, SARIMA, and TimesFM are statistically indistinguishable, and the ranking we ourselves reported on a shorter series reversed when more data arrived, illustrating that a gap smaller than the confidence interval is not a result. Gradient boosting over lags is detectably worse. The variable that moved forecast accuracy was the length of the training series, by roughly an order of magnitude more than the choice of model. The foundation model's defensible advantage is not accuracy but insensitivity to short history, which matters precisely in the surveillance settings where many series must be forecast and none of them is long. For a single long-running indicator, the least expensive model is sufficient, and saying so requires having measured the uncertainty.

---

## Data and Code Availability

All code, extraction scripts, aggregated series, benchmark outputs, and uncertainty results are publicly available at:
https://github.com/fabianofilho/cardiovascular-timeseries-prediction

Individual death records are not redistributed; the extraction script reproduces them from the public DataSUS source.

---

## Acknowledgments

> **TODO:** Add acknowledgments: funding sources, lab affiliation, computational resources.

---

## Conflict of Interest

The authors declare no conflicts of interest.

---

## References

1. Roth GA, Mensah GA, Johnson CO, et al. Global Burden of Cardiovascular Diseases and Risk Factors, 1990-2019: Update From the GBD 2019 Study. *Journal of the American College of Cardiology*. 2020;76(25):2982-3021. doi:10.1016/j.jacc.2020.11.010

2. Oliveira GMM, Brant LCC, Polanczyk CA, et al. Cardiovascular Statistics, Brazil 2020. *Arquivos Brasileiros de Cardiologia*. 2020;115(3):308-439. doi:10.36660/abc.20200812

3. Brant LCC, Nascimento BR, Passos VMA, et al. Variations and particularities in cardiovascular disease mortality in Brazil and Brazilian states in 1990 and 2015: estimates from the Global Burden of Disease. *Revista Brasileira de Epidemiologia*. 2017;20(Suppl 1):116-128. doi:10.1590/1980-5497201700050010

4. Mansur AP, Favarato D. Mortality due to cardiovascular diseases in Brazil and in the metropolitan region of São Paulo: a 2011 update. *Arquivos Brasileiros de Cardiologia*. 2012;99(2):755-761.

5. Ribeiro AL, Duncan BB, Brant LC, Lotufo PA, Mill JG, Barreto SM. Cardiovascular Health in Brazil: Trends and Perspectives. *Circulation*. 2016;133(4):422-433. doi:10.1161/CIRCULATIONAHA.114.008727

6. Brant LCC, Nascimento BR, Ribeiro ALP, et al. Cardiovascular diseases mortality in Brazilian municipalities: estimates from the Global Burden of Disease study, 2000-2018. *The Lancet Regional Health, Americas*. 2025;45:101164. doi:10.1016/j.lana.2025.101164

7. Box GEP, Jenkins GM, Reinsel GC, Ljung GM. *Time Series Analysis: Forecasting and Control*. 5th ed. John Wiley & Sons; 2015.

8. Taylor SJ, Letham B. Forecasting at scale. *The American Statistician*. 2018;72(1):37-45. doi:10.1080/00031305.2017.1380080

9. Das A, Kong W, Sen R, Zhou Y. A decoder-only foundation model for time-series forecasting. In: *Proceedings of the 41st International Conference on Machine Learning (ICML)*. Vienna, Austria. PMLR 235; 2024. arXiv:2310.10688

10. Ministério da Saúde. Sistema de Informações sobre Mortalidade (SIM/DataSUS). Available from: http://www.datasus.gov.br/

11. World Health Organization. International Statistical Classification of Diseases and Related Health Problems, 10th Revision (ICD-10). Available from: https://www.who.int/standards/classifications/classification-of-diseases

12. Analitis A, Katsouyanni K, Biggeri A, et al. Effects of cold weather on mortality: results from 15 European cities within the PHEWE project. *American Journal of Epidemiology*. 2008;168(12):1397-1408. doi:10.1093/aje/kwn266

13. Phung D, Thai PK, Guo Y, Morawska L, Rutherford S, Chu C. Ambient temperature and risk of cardiovascular hospitalization: An updated systematic review and meta-analysis. *Science of The Total Environment*. 2016;550:1084-1102.

14. Gupta A, et al. MIRA: Medical Time Series Foundation Model for Real-World Health Data. In: *Advances in Neural Information Processing Systems (NeurIPS)*; 2025. arXiv:2506.07584

15. Ansari AF, Stella L, Turkmen C, et al. Chronos: Learning the Language of Time Series. arXiv:2403.07815; 2024.

16. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785-794. doi:10.1145/2939672.2939785

17. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. In: *Advances in Neural Information Processing Systems (NeurIPS)*; 2018. arXiv:1706.09516

18. Diebold FX, Mariano RS. Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*. 1995;13(3):253-263. doi:10.1080/07350015.1995.10524599

19. Harvey D, Leybourne S, Newbold P. Testing the equality of prediction mean squared errors. *International Journal of Forecasting*. 1997;13(2):281-291. doi:10.1016/S0169-2070(96)00719-4

20. Instituto Nacional de Meteorologia (INMET). Dados históricos anuais, estações automáticas. Available from: https://portal.inmet.gov.br/dadoshistoricos
