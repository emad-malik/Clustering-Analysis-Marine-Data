# Vessel Behavior Clustering and Anomaly Detection

This project analyzes vessel movement patterns using AIS data scraped from MarineTraffic. It applies clustering algorithms to spatial data and detects anomalous behavior using statistical techniques. 

## Dataset Overview

The dataset contains vessel-level attributes such as:

- LAT, LON – Geographic position  
- SPEED, COURSE, HEADING – Navigation behavior  
- SHIPTYPE, SHIP_ID, LENGTH, WIDTH – Ship metadata  
- STATUS_NAME, DESTINATION, etc.

## 1. Data Cleaning

Cleaning steps included:
- Dropping columns with excessive null values
- Removing or imputing incomplete rows
- Converting data types (e.g., SPEED to float)
- Normalizing numerical fields where necessary
- Filtering out clearly invalid or nonsensical entries (e.g., negative speeds)

Final cleaned features used:
- LAT, LON, SPEED, COURSE, HEADING, LENGTH, WIDTH, SHIPTYPE, SHIP_ID

## 2. Vessel Clustering

Objective: Group vessels by spatial behavior using positional data

Method: K-Means Clustering  
- Input: Latitude and Longitude (scaled with StandardScaler)  
- Output: A new column 'POSITION_CLUSTER' indicating the spatial group

Evaluation Metrics:
- Inertia (within-cluster sum of squares)
- Silhouette Score (higher is better)
- Davies-Bouldin Index (lower is better)

Example:

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

print("Inertia:", kmeans.inertia_)
print("Silhouette:", silhouette_score(X_scaled, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_scaled, labels))
````

## 3. Anomaly Detection

Objective: Identify vessels behaving in a way that deviates significantly from typical patterns based on speed, course, and location.

### 3.1 Speed-Based Anomalies

Ships are grouped by `SHIPTYPE`, and their speed distribution is used to calculate z-scores. Vessels with extremely high or low speeds relative to their type are flagged.

Method:

* Compute z-score of SPEED within each SHIPTYPE
* Threshold: |z| > 2.5 for strict filtering, |z| > 0.2 for exploratory analysis

Example:

```python
df['z_speed'] = df.groupby('SHIPTYPE')['SPEED'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0)
)
speed_anomalies = df[df['z_speed'].abs() > z_thresh]
```

### 3.2 Course Drift (Planned)

Ships are expected to maintain a consistent relationship between `COURSE` and `HEADING`. Vessels with large, unexplained deviations may be off-route or performing evasive maneuvers.

Potential method:

* Compute absolute difference between HEADING and COURSE
* Flag if deviation exceeds a threshold (e.g., > 45°)

### 3.3 Suspicious Stops (Planned)

Ships that remain stationary (SPEED ≈ 0) for long durations outside known ports or anchor zones may indicate illegal fishing, smuggling, or distress.

Potential method:

* Identify sequences of zero/near-zero speed
* Cross-reference LAT/LON with known port regions or protected zones
* Flag if stop occurs in unusual region

## Tools and Techniques

* Web scraping: Selenium + Undetected Browser
* Data manipulation: Pandas, NumPy
* Clustering: scikit-learn (KMeans)
* Outlier detection: Z-score based filtering
* Visualization: Seaborn, Matplotlib
* Notebook environment: Colab
