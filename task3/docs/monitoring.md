\# Monitoring and Alerting



\## Overview



The Olist Late Delivery Prediction API is monitored using Prometheus metrics and application logs.



The monitoring setup tracks:



\- Request count

\- Request latency

\- Prediction error count

\- Prediction distribution

\- Prediction logs



\## Metrics



\### 1. Request Count



Metric:



`http\_requests\_total`



This metric tracks the number of HTTP requests received by the API, including the HTTP method, endpoint, and response status.



It can be used to monitor API traffic and detect unusual changes in request volume.



\### 2. Request Latency



Metric:



`http\_request\_duration\_seconds`



This metric tracks the time required to process API requests.



It can be used to detect slow responses and performance degradation.



\### 3. Prediction Errors



Metric:



`prediction\_errors\_total`



This metric counts failed prediction requests.



The error rate can be calculated as:



`prediction error rate = prediction errors / total prediction requests`



A sustained increase in prediction errors should trigger an investigation.



\### 4. Prediction Distribution



Metric:



`prediction\_total`



This metric tracks the number of predictions for each prediction result:



\- `is\_late="0"` — predicted on time

\- `is\_late="1"` — predicted late



The distribution should be monitored over time to detect unusual changes in model behavior.



\## Prediction Logs



Prediction requests and results are stored in:



`logs/app.log`



Each successful prediction log includes information such as:



\- Prediction result

\- Prediction probability

\- Prediction latency

\- Model name

\- Model version



Rejected and unexpected prediction errors are also logged.



These logs can be used later to compare predictions with the actual delivery outcome when the real delivery date becomes available.



\## Drift Monitoring



Prediction distribution should be monitored over time.



A significant and sustained change in the proportion of:



\- `is\_late=0`

\- `is\_late=1`



may indicate a change in the data or in the behavior of the model.



If sufficient historical data becomes available, additional statistical drift detection can be added to compare recent input features with the training data distribution.



\## Alerting Decisions



The following conditions should trigger an alert or investigation:



\### Alert 1 — High Prediction Error Rate



Trigger when the prediction error rate remains above an agreed threshold for a sustained period.



Purpose:



Detect API or model-serving problems that prevent successful predictions.



\### Alert 2 — High API Latency



Trigger when API request latency becomes significantly higher than the normal baseline for a sustained period.



Purpose:



Detect performance degradation affecting prediction response time.



\### Alert 3 — Prediction Distribution Change



Trigger when the proportion of late predictions changes significantly from the established baseline.



Purpose:



Detect possible data distribution changes or model behavior changes.



\### Alert 4 — API Availability Problem



Trigger when the health endpoint fails or the API becomes unavailable.



Purpose:



Detect service outages before they affect users for an extended period.



\## Future Monitoring Improvements



The current implementation provides the core monitoring metrics and prediction logging required for the service.



Future improvements could include:



\- Prometheus alerting rules

\- Grafana dashboards

\- Automated data drift detection

\- Automated model performance monitoring

\- Comparing predictions with actual delivery outcomes after delivery

\- Monitoring model performance metrics such as precision, recall, and F1-score

