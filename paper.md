# Forecasting Cardiovascular Mortality in São Paulo State Using Foundation Models: A Comparative Study of SARIMA, Prophet, and TimesFM

**Fabiano B. N. Filho¹ · Isabela Venancio da Silva¹ · Alexandre Chiavegatto Dias Porto Filho¹**

¹ Faculdade de Saúde Pública, Universidade de São Paulo (FSP-USP)

---

## Abstract

Cardiovascular diseases (CVDs) remain the leading cause of death in Brazil, accounting for approximately 30% of all deaths. Accurate forecasting of cardiovascular mortality trends is essential for public health planning and resource allocation. In recent years, foundation models pre-trained on large-scale time series corpora have shown strong zero-shot performance across diverse forecasting tasks, yet their application to epidemiological surveillance using real-world public health data remains largely unexplored.

In this study, we evaluate the performance of three time series forecasting models — Seasonal ARIMA (SARIMA), Prophet, and TimesFM — in predicting monthly cardiovascular mortality in the state of São Paulo, Brazil. Using 60 months of real-world data from the Brazilian Mortality Information System (SIM/DataSUS, January 2019 to December 2023), we conducted a rolling origin cross-validation with a 6-month forecast horizon, generating 186 predictions per model.

TimesFM, a decoder-only foundation model, achieved the best performance across all metrics (MAE = 586.02, RMSE = 765.77, sMAPE = 7.30%), followed closely by SARIMA (sMAPE = 7.45%) and Prophet (sMAPE = 7.68%). A clear seasonal pattern was observed, with mortality peaks during winter months (June–July) and troughs in summer (February), consistent with established evidence on temperature–cardiovascular mortality associations. These findings highlight the potential of foundation models for epidemiological time series forecasting and provide a baseline for future studies incorporating exogenous variables and finer spatial granularity.

**Keywords:** cardiovascular mortality, time series forecasting, foundation models, TimesFM, SARIMA, Prophet, DataSUS, epidemiological surveillance, Brazil

---

## 1. Introduction

Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible for approximately 17.9 million deaths per year [1]. In Brazil, CVDs have been the primary cause of mortality since the 1960s, accounting for roughly 30% of all deaths in recent decades — surpassed only by COVID-19 in 2020 and 2021 [2, 3]. The state of São Paulo, the most populous in Brazil with over 46 million inhabitants, bears a substantial share of this burden [4].

Despite a sustained decline in age-standardized CVD mortality rates since the 1990s — attributed to improved risk factor control, expanded primary care coverage, and advances in acute treatment [5] — the absolute number of cardiovascular deaths continues to rise due to population growth and aging [6]. This underscores the ongoing need for robust forecasting tools that can anticipate mortality trends and inform resource allocation, particularly within Brazil's universal public health system (SUS).

Time series forecasting has a long history in epidemiological surveillance. Classical statistical models such as Seasonal ARIMA (SARIMA) have been widely used for modeling disease incidence and mortality [7]. More recently, Meta's Prophet model has gained adoption due to its user-friendly decomposition of trend and seasonality components [8]. However, both approaches require per-series model fitting and parameter tuning, which limits scalability across multiple health indicators and geographic units.

The emergence of foundation models for time series forecasting represents a paradigm shift in the field. These models, pre-trained on massive and diverse time series corpora, can produce competitive zero-shot forecasts on previously unseen datasets without task-specific training [9]. TimesFM, a decoder-only transformer model developed by Google Research with 200 million parameters pre-trained on approximately 100 billion real-world time points, has demonstrated state-of-the-art zero-shot performance across multiple benchmarks [9]. Despite this promise, the application of foundation models to public health time series — particularly using administrative mortality data from low- and middle-income countries — remains largely unexplored.

This study addresses this gap by comparing the forecasting performance of SARIMA, Prophet, and TimesFM on a monthly cardiovascular mortality series derived from real SIM/DataSUS records for the state of São Paulo (2019–2023). We employ a rigorous rolling origin cross-validation framework and evaluate predictions across three standard metrics.

---

## 2. Methods

### 2.1 Data Source and Preprocessing

This study utilized individual death records from the Brazilian Mortality Information System (Sistema de Informações sobre Mortalidade — SIM), maintained by the Ministry of Health and accessible through the DataSUS platform [10]. Records were extracted for the state of São Paulo (UF = SP) covering the period from January 2019 to December 2023. Deaths were filtered by underlying cause of death using ICD-10 codes I00–I99 (diseases of the circulatory system) [11].

From a total of 1,775,800 raw mortality records, 461,950 were classified as cardiovascular deaths. These were aggregated into a monthly time series of 60 data points. The resulting series had a mean of approximately 7,700 deaths per month, with a minimum of 5,978 (April 2020) and a maximum of 9,582 (January 2022).

> **TODO:** Consider adding a brief note on data quality, e.g., completeness of death registration in SP.

### 2.2 Forecasting Models

Three forecasting models were evaluated:

- **SARIMA:** A Seasonal Autoregressive Integrated Moving Average model with order (1,1,1) and seasonal order (0,1,1,12), implementing yearly seasonality through seasonal differencing [7].

- **Prophet:** An additive decomposition model developed by Meta, configured with yearly seasonality enabled and weekly/daily seasonalities disabled, appropriate for monthly aggregated data [8].

- **TimesFM:** A decoder-only foundation model for time series forecasting [9]. We used the publicly available `google/timesfm-2.5-200m-pytorch` checkpoint with a context window of 512 time points. The model was applied in a zero-shot setting without any fine-tuning on the target series.

### 2.3 Backtesting Strategy

Model performance was evaluated using rolling origin cross-validation, a temporal validation strategy that simulates real-world forecasting conditions by iteratively expanding the training window and generating out-of-sample predictions. The configuration was as follows:

| Parameter | Value |
|---|---|
| Forecast horizon | 6 months |
| Minimum training window | 24 months |
| Total rolling windows | 31 |
| Total predictions per model | 186 (31 × 6) |

### 2.4 Evaluation Metrics

Predictive accuracy was assessed using three complementary metrics:

- **MAE (Mean Absolute Error):** Average absolute difference between predicted and observed values, providing an interpretable measure of typical forecast error in deaths/month.
- **RMSE (Root Mean Squared Error):** Square root of the mean squared error, penalizing larger deviations more heavily than MAE.
- **sMAPE (Symmetric Mean Absolute Percentage Error):** A scale-independent percentage error metric that is symmetric with respect to over- and under-prediction, suitable for comparing forecasting performance across different series.

---

## 3. Results

### 3.1 Comparative Model Performance

The results of the rolling origin backtesting are summarized below. TimesFM outperformed both SARIMA and Prophet across all three evaluation metrics.

**Table 1.** Comparative performance of forecasting models for monthly cardiovascular mortality in São Paulo, Brazil (2019–2023). All models were evaluated on the same 186 out-of-sample predictions from 31 rolling origin windows with a 6-month horizon.

| Model | MAE | RMSE | sMAPE (%) | n |
|---|---|---|---|---|
| **TimesFM** | **586.02** | **765.77** | **7.30** | 186 |
| SARIMA | 603.51 | 787.83 | 7.45 | 186 |
| Prophet | 613.80 | 767.23 | 7.68 | 186 |

TimesFM achieved the lowest MAE (586.02), RMSE (765.77), and sMAPE (7.30%). Given an average monthly mortality of approximately 8,000 deaths, an MAE of 586 corresponds to a relative error of approximately 7.3%. SARIMA performed as a competitive baseline (sMAPE = 7.45%), with Prophet ranking third (sMAPE = 7.68%). The difference in sMAPE between the best and worst models was only 0.38 percentage points, indicating that the underlying series exhibits strong and relatively predictable seasonal and trend components.

TimesFM's predicted range [6,664–9,078] was more conservative than the observed range [6,217–9,582], suggesting the model tends to slightly underestimate the amplitude of extreme events.

### 3.2 Seasonality

The cardiovascular mortality series exhibited a clear seasonal pattern, with peaks during winter months (June–July) and troughs in summer (February). This pattern was consistent across all five years of the study period and is in line with established evidence on the association between lower ambient temperatures and increased cardiovascular events [12, 13].

> **TODO:** Add Figure 1 — Monthly cardiovascular mortality in São Paulo state (January 2019 – December 2023), showing the seasonal pattern with winter peaks and summer troughs. Replace placeholder with actual figure from the repository notebooks.

> **TODO:** Add Figure 2 — Comparison of observed versus predicted cardiovascular mortality for SARIMA, Prophet, and TimesFM across the rolling origin backtesting period. Replace placeholder with actual figure from the repository notebooks.

---

## 4. Discussion

This study provides, to our knowledge, the first comparison of a foundation model (TimesFM) against classical and modern statistical approaches for forecasting cardiovascular mortality using real administrative data from Brazil's public health system. The results demonstrate that TimesFM achieved the best performance in a zero-shot setting, without requiring any training on the target series, while classical SARIMA remained a strong competitor.

### 4.1 Foundation Models in Epidemiological Forecasting

> **TODO:** Develop this section. Key points to address:

- TimesFM's zero-shot capability eliminates the need for per-series model selection and hyperparameter tuning, which is operationally significant for health surveillance agencies that need to monitor hundreds of mortality indicators simultaneously.
- Compare with other recent applications of foundation models in health settings (e.g., MIRA [14], Chronos [15]).
- Discuss the decoder-only architecture and the patching mechanism as enabling factors for generalization to unseen health time series.
- Position within the broader trend of foundation models transitioning from NLP to structured data domains.

### 4.2 Competitiveness of Classical Models

> **TODO:** Develop this section. Key points to address:

- The narrow performance gap (ΔsMAPE ≈ 0.38 pp) between TimesFM and SARIMA suggests that the series has strong autoregressive and seasonal structure that simpler models can capture effectively.
- Discuss when SARIMA may be preferable: low-resource settings, interpretability needs, transparency for health policy communication.
- Prophet's decomposition of trend + seasonality offers unique value for understanding the driving forces behind mortality patterns, even if it ranked third on aggregate metrics.
- Consider a cost-benefit analysis: computational cost of running TimesFM vs. marginal accuracy gain.

### 4.3 Seasonality and Public Health Implications

> **TODO:** Develop this section. Key points to address:

- The observed winter mortality peak aligns with evidence on cold-related cardiovascular risk, including increased blood pressure, platelet aggregation, and respiratory infections that can trigger cardiovascular events [12].
- Discuss implications for seasonal resource planning: ICU bed allocation, emergency services, medication supply chains.
- The COVID-19 period (2020–2021) introduced anomalies (e.g., minimum of 5,978 in April 2020) that may reflect delayed diagnosis, avoidance of emergency care, or competing causes of death. Discuss how models handled this disruption.
- São Paulo's subtropical climate results in milder winters than temperate regions, yet the seasonal effect on CVD mortality is still pronounced — discuss implications for tropical and subtropical settings.

### 4.4 Data Quality and SIM/DataSUS Considerations

> **TODO:** Develop this section. Key points to address:

- São Paulo has one of the highest death registration completeness rates in Brazil (>95%), which strengthens the reliability of the series.
- Discuss potential issues: garbage codes in ICD-10 classification, redistribution algorithms used in GBD studies vs. raw SIM data.
- The series uses raw SIM data without garbage code redistribution — discuss implications for comparability with GBD estimates.
- Consider data latency: SIM records may take months to consolidate, affecting real-time forecasting applications.

### 4.5 Limitations

> **TODO:** Develop this section. Key points to address:

- **Univariate analysis:** no exogenous variables (temperature, air pollution, APS coverage, vaccination rates, socioeconomic indicators).
- **Single geographic unit:** state-level aggregation masks within-state heterogeneity across municipalities and DRS (Departamentos Regionais de Saúde).
- **Short series:** 60 monthly observations limits the complexity of seasonal patterns that can be captured and the number of backtesting windows.
- **Zero-shot only:** TimesFM was not fine-tuned — fine-tuning on Brazilian health data could potentially improve performance.
- **Pandemic period:** the study period includes COVID-19, which introduced unprecedented mortality patterns that may affect model evaluation.

### 4.6 Future Directions

> **TODO:** Develop this section. Key points to address:

- **Exogenous variables:** daily/monthly temperature, air quality (PM₂.₅), primary care coverage indicators (e.g., proportion of population covered by Estratégia Saúde da Família), population aging metrics.
- **Stratification:** replicate the analysis by sex, age group (40–59, 60–79, 80+), ICD-10 subcategory (ischemic heart disease I20–I25, cerebrovascular disease I60–I69), and geographic unit (municipality, DRS, micro-region).
- **Model ensembling:** investigate whether combining TimesFM, SARIMA, and Prophet predictions improves robustness.
- **Fine-tuning:** evaluate whether domain-specific fine-tuning of TimesFM on Brazilian health time series improves performance beyond zero-shot.
- **Operational integration:** discuss pathways for integrating forecasting models into existing health surveillance platforms (e.g., e-SUS, SIVEP-Gripe).
- **Temporal scope:** include SIH (hospitalization) data alongside SIM (mortality) to model the full care cascade from incidence to death.

---

## 5. Conclusion

This study evaluated three time series forecasting approaches — SARIMA, Prophet, and TimesFM — for predicting monthly cardiovascular mortality in São Paulo state using real data from Brazil's Mortality Information System. TimesFM, a foundation model applied in a zero-shot setting, achieved the best performance across all evaluated metrics, demonstrating that pre-trained foundation models can effectively forecast epidemiological time series without domain-specific training. The narrow performance gap with SARIMA highlights the strong predictability of the series and the continued relevance of classical statistical methods. The consistent seasonal pattern of higher winter mortality reinforces the need for anticipatory public health strategies during colder months. Future work should incorporate exogenous variables, finer spatial and demographic stratification, and operational integration to fully realize the potential of foundation models for health surveillance in Brazil.

---

## Data and Code Availability

All code, data extraction scripts, and results are publicly available at:
https://github.com/fabianofilho/cardiovascular-timeseries-prediction

---

## Acknowledgments

> **TODO:** Add acknowledgments: funding sources, lab/supervisor, computational resources.

---

## Conflict of Interest

The authors declare no conflicts of interest.

---

## References

1. Roth GA, Mensah GA, Johnson CO, et al. Global Burden of Cardiovascular Diseases and Risk Factors, 1990–2019: Update From the GBD 2019 Study. *Journal of the American College of Cardiology*. 2020;76(25):2982–3021. doi:10.1016/j.jacc.2020.11.010

2. Oliveira GMM, Brant LCC, Polanczyk CA, et al. Cardiovascular Statistics – Brazil 2020. *Arquivos Brasileiros de Cardiologia*. 2020;115(3):308–439. doi:10.36660/abc.20200812

3. Brant LCC, Nascimento BR, Passos VMA, et al. Variations and particularities in cardiovascular disease mortality in Brazil and Brazilian states in 1990 and 2015: estimates from the Global Burden of Disease. *Revista Brasileira de Epidemiologia*. 2017;20(Suppl 1):116–128. doi:10.1590/1980-5497201700050010

4. Mansur AP, Favarato D. Mortality due to cardiovascular diseases in Brazil and in the metropolitan region of São Paulo: a 2011 update. *Arquivos Brasileiros de Cardiologia*. 2012;99(2):755–761.

5. Ribeiro AL, Duncan BB, Brant LC, Lotufo PA, Mill JG, Barreto SM. Cardiovascular Health in Brazil: Trends and Perspectives. *Circulation*. 2016;133(4):422–433. doi:10.1161/CIRCULATIONAHA.114.008727

6. Brant LCC, Nascimento BR, Ribeiro ALP, et al. Cardiovascular diseases mortality in Brazilian municipalities: estimates from the Global Burden of Disease study, 2000–2018. *The Lancet Regional Health – Americas*. 2025;45:101164. doi:10.1016/j.lana.2025.101164

7. Box GEP, Jenkins GM, Reinsel GC, Ljung GM. *Time Series Analysis: Forecasting and Control*. 5th ed. John Wiley & Sons; 2015.

8. Taylor SJ, Letham B. Forecasting at scale. *The American Statistician*. 2018;72(1):37–45. doi:10.1080/00031305.2017.1380080

9. Das A, Kong W, Sen R, Zhou Y. A decoder-only foundation model for time-series forecasting. In: *Proceedings of the 41st International Conference on Machine Learning (ICML)*. Vienna, Austria. PMLR 235; 2024. arXiv:2310.10688

10. Ministério da Saúde. Sistema de Informações sobre Mortalidade (SIM/DataSUS). Available from: http://www.datasus.gov.br/. Accessed: 2024.

11. World Health Organization. International Statistical Classification of Diseases and Related Health Problems, 10th Revision (ICD-10). Available from: https://www.who.int/standards/classifications/classification-of-diseases

12. Analitis A, Katsouyanni K, Biggeri A, et al. Effects of cold weather on mortality: results from 15 European cities within the PHEWE project. *American Journal of Epidemiology*. 2008;168(12):1397–1408. doi:10.1093/aje/kwn266

13. Phung D, Thai PK, Guo Y, Morawska L, Rutherford S, Chu C. Ambient temperature and risk of cardiovascular hospitalization: An updated systematic review and meta-analysis. *Science of The Total Environment*. 2016;550:1084–1102.

14. Gupta A, et al. MIRA: Medical Time Series Foundation Model for Real-World Health Data. In: *Advances in Neural Information Processing Systems (NeurIPS)*; 2025. arXiv:2506.07584

15. Ansari AF, Stella L, Turkmen C, et al. Chronos: Learning the Language of Time Series. arXiv:2403.07815; 2024.
