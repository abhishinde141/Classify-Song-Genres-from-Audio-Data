# Classify-Song-Genres-from-Audio-Data
## 1. Preparing our dataset
<p><em>These recommendations are so on point! How does this playlist know me so well?</em></p>
<p>Over the past few years, streaming services with huge catalogs have become the primary means through which most people listen to their favorite music. But at the same time, the sheer amount of music on offer can mean users might be a bit overwhelmed when trying to look for newer music that suits their tastes.</p>
<p>For this reason, streaming services have looked into means of categorizing music to allow for personalized recommendations. One method involves direct analysis of the raw audio information in a given song, scoring the raw data on a variety of metrics. Today, we'll be examining data compiled by a research group known as The Echo Nest. Our goal is to look through this dataset and classify songs as being either 'Hip-Hop' or 'Rock' - all without listening to a single one ourselves. In doing so, we will learn how to clean our data, do some exploratory data visualization, and use feature reduction towards the goal of feeding our data through some simple machine learning algorithms, such as decision trees and logistic regression.</p>
<p>To begin with, let's load the metadata about our tracks alongside the track metrics compiled by The Echo Nest. A song is about more than its title, artist, and number of listens. We have another dataset that has musical features of each track such as <code>danceability</code> and <code>acousticness</code> on a scale from -1 to 1. These exist in two different files, which are in different formats - CSV and JSON. While CSV is a popular file format for denoting tabular data, JSON is another common file format in which databases often return the results of a given query.</p>
<p>Let's start by creating two pandas <code>DataFrames</code> out of these files that we can merge so we have features and labels (often also referred to as <code>X</code> and <code>y</code>) for the classification later on.</p>

#### Task 1: Instructions
Read in the data using pandas and merge the DataFrames into one usable dataset.

* Using the pandas read_csv() function, read in the file with the track metadata (datasets/fma-rock-vs-hiphop.csv) and name the DataFrame tracks.

* Using the pandas read_json() function, read in the JSON file with the track acoustic metrics (datasets/echonest-metrics.json) and name the DataFrame echonest_metrics. Set the precise_float argument to True when reading in your data.

* Merge the DataFrames on matching track_id values. Only retain the track_id and genre_top columns of tracks. echonest_metrics should be the first (left) data frame in the merge.

* Inspect the DataFrame using the .info() method.


```python
import pandas as pd

# Read in the track metadata with genre labels from CSV
tracks = pd.read_csv('datasets/fma-rock-vs-hiphop.csv')

# Read in the track metrics with the features from JSON, setting precise_float to True
echonest_metrics = pd.read_json('datasets/echonest-metrics.json', precise_float=True)


# Merge the DataFrames on the track_id column, with echonest_metrics as the left DataFrame
echo_tracks = echonest_metrics.merge(tracks[['track_id', 'genre_top']], on='track_id')

# Inspect the DataFrame
print("DataFrame Information:")
print(echo_tracks.info())


```

    DataFrame Information:
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4802 entries, 0 to 4801
    Data columns (total 10 columns):
    acousticness        4802 non-null float64
    danceability        4802 non-null float64
    energy              4802 non-null float64
    instrumentalness    4802 non-null float64
    liveness            4802 non-null float64
    speechiness         4802 non-null float64
    tempo               4802 non-null float64
    track_id            4802 non-null int64
    valence             4802 non-null float64
    genre_top           4802 non-null object
    dtypes: float64(8), int64(1), object(1)
    memory usage: 412.7+ KB
    None



```python
%%nose

def test_tracks_read():
    try:
        pd.testing.assert_frame_equal(tracks, pd.read_csv('datasets/fma-rock-vs-hiphop.csv'))
    except AssertionError:
        assert False, "The tracks data frame was not read in correctly."

def test_metrics_read():
    ech_met_test = pd.read_json('datasets/echonest-metrics.json', precise_float=True)
    try:
        pd.testing.assert_frame_equal(echonest_metrics, ech_met_test)
    except AssertionError:
        assert False, "The echonest_metrics data frame was not read in correctly."
        
def test_merged_shape(): 
    merged_test = echonest_metrics.merge(tracks[['genre_top', 'track_id']], on='track_id')
    try:
        pd.testing.assert_frame_equal(echo_tracks, merged_test)
    except AssertionError:
        assert False, ('The two datasets should be merged on matching track_id values '
                       'keeping only the track_id and genre_top columns of tracks.')
```






    3/3 tests passed




## 2. Pairwise relationships between continuous variables
<p>We typically want to avoid using variables that have strong correlations with each other -- hence avoiding feature redundancy -- for a few reasons:</p>
<ul>
<li>To keep the model simple and improve interpretability (with many features, we run the risk of overfitting).</li>
<li>When our datasets are very large, using fewer features can drastically speed up our computation time.</li>
</ul>
<p>To get a sense of whether there are any strongly correlated features in our data, we will use built-in functions in the <code>pandas</code> package.</p>

#### Task 2: Instructions
Explore correlations in our dataset using pandas corr function.

* Visually inspect the correlation table generated from DataFrame.corr() for any strong correlations.


```python
# Create a correlation matrix

corr_metrics = echo_tracks.corr()

# Display the correlation matrix with a color gradient for better visualization
corr_metrics.style.background_gradient()
```




<style  type="text/css" >
    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col0 {
            background-color:  #023858;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col1 {
            background-color:  #e0dded;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col2 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col3 {
            background-color:  #97b7d7;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col4 {
            background-color:  #f3edf5;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col5 {
            background-color:  #b8c6e0;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col6 {
            background-color:  #e1dfed;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col7 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col8 {
            background-color:  #e2dfee;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col0 {
            background-color:  #d0d1e6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col1 {
            background-color:  #023858;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col2 {
            background-color:  #fbf3f9;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col3 {
            background-color:  #f3edf5;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col4 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col5 {
            background-color:  #80aed2;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col6 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col7 {
            background-color:  #bdc8e1;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col8 {
            background-color:  #529bc7;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col0 {
            background-color:  #f5eff6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col1 {
            background-color:  #fef6fa;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col2 {
            background-color:  #023858;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col3 {
            background-color:  #c4cbe3;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col4 {
            background-color:  #dcdaeb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col5 {
            background-color:  #dedcec;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col6 {
            background-color:  #adc1dd;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col7 {
            background-color:  #a7bddb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col8 {
            background-color:  #d9d8ea;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col0 {
            background-color:  #97b7d7;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col1 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col2 {
            background-color:  #d2d3e7;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col3 {
            background-color:  #023858;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col4 {
            background-color:  #fdf5fa;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col5 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col6 {
            background-color:  #d9d8ea;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col7 {
            background-color:  #f4eef6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col8 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col0 {
            background-color:  #ced0e6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col1 {
            background-color:  #ede8f3;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col2 {
            background-color:  #bdc8e1;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col3 {
            background-color:  #dbdaeb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col4 {
            background-color:  #023858;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col5 {
            background-color:  #c0c9e2;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col6 {
            background-color:  #dcdaeb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col7 {
            background-color:  #bdc8e1;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col8 {
            background-color:  #e8e4f0;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col0 {
            background-color:  #b8c6e0;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col1 {
            background-color:  #93b5d6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col2 {
            background-color:  #eae6f1;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col3 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col4 {
            background-color:  #eae6f1;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col5 {
            background-color:  #023858;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col6 {
            background-color:  #dbdaeb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col7 {
            background-color:  #d0d1e6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col8 {
            background-color:  #bfc9e1;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col0 {
            background-color:  #d0d1e6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col1 {
            background-color:  #fef6fa;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col2 {
            background-color:  #a7bddb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col3 {
            background-color:  #c5cce3;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col4 {
            background-color:  #f0eaf4;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col5 {
            background-color:  #c8cde4;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col6 {
            background-color:  #023858;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col7 {
            background-color:  #d0d1e6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col8 {
            background-color:  #d6d6e9;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col0 {
            background-color:  #fff7fb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col1 {
            background-color:  #d2d2e7;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col2 {
            background-color:  #b5c4df;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col3 {
            background-color:  #f5eef6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col4 {
            background-color:  #e9e5f1;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col5 {
            background-color:  #d1d2e6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col6 {
            background-color:  #e1dfed;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col7 {
            background-color:  #023858;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col8 {
            background-color:  #dedcec;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col0 {
            background-color:  #cdd0e5;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col1 {
            background-color:  #4c99c5;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col2 {
            background-color:  #d1d2e6;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col3 {
            background-color:  #efe9f3;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col4 {
            background-color:  #f7f0f7;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col5 {
            background-color:  #a5bddb;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col6 {
            background-color:  #d3d4e7;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col7 {
            background-color:  #c6cce3;
        }    #T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col8 {
            background-color:  #023858;
        }</style>  
<table id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736a" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >acousticness</th> 
        <th class="col_heading level0 col1" >danceability</th> 
        <th class="col_heading level0 col2" >energy</th> 
        <th class="col_heading level0 col3" >instrumentalness</th> 
        <th class="col_heading level0 col4" >liveness</th> 
        <th class="col_heading level0 col5" >speechiness</th> 
        <th class="col_heading level0 col6" >tempo</th> 
        <th class="col_heading level0 col7" >track_id</th> 
        <th class="col_heading level0 col8" >valence</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row0" class="row_heading level0 row0" >acousticness</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col0" class="data row0 col0" >1</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col1" class="data row0 col1" >-0.0289537</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col2" class="data row0 col2" >-0.281619</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col3" class="data row0 col3" >0.19478</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col4" class="data row0 col4" >-0.0199914</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col5" class="data row0 col5" >0.072204</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col6" class="data row0 col6" >-0.0263097</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col7" class="data row0 col7" >-0.372282</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow0_col8" class="data row0 col8" >-0.0138406</td> 
    </tr>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row1" class="row_heading level0 row1" >danceability</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col0" class="data row1 col0" >-0.0289537</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col1" class="data row1 col1" >1</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col2" class="data row1 col2" >-0.242032</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col3" class="data row1 col3" >-0.255217</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col4" class="data row1 col4" >-0.106584</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col5" class="data row1 col5" >0.276206</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col6" class="data row1 col6" >-0.242089</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col7" class="data row1 col7" >0.0494541</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow1_col8" class="data row1 col8" >0.473165</td> 
    </tr>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row2" class="row_heading level0 row2" >energy</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col0" class="data row2 col0" >-0.281619</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col1" class="data row2 col1" >-0.242032</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col2" class="data row2 col2" >1</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col3" class="data row2 col3" >0.0282377</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col4" class="data row2 col4" >0.113331</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col5" class="data row2 col5" >-0.109983</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col6" class="data row2 col6" >0.195227</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col7" class="data row2 col7" >0.140703</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow2_col8" class="data row2 col8" >0.0386027</td> 
    </tr>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row3" class="row_heading level0 row3" >instrumentalness</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col0" class="data row3 col0" >0.19478</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col1" class="data row3 col1" >-0.255217</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col2" class="data row3 col2" >0.0282377</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col3" class="data row3 col3" >1</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col4" class="data row3 col4" >-0.0910218</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col5" class="data row3 col5" >-0.366762</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col6" class="data row3 col6" >0.022215</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col7" class="data row3 col7" >-0.275623</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow3_col8" class="data row3 col8" >-0.219967</td> 
    </tr>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row4" class="row_heading level0 row4" >liveness</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col0" class="data row4 col0" >-0.0199914</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col1" class="data row4 col1" >-0.106584</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col2" class="data row4 col2" >0.113331</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col3" class="data row4 col3" >-0.0910218</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col4" class="data row4 col4" >1</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col5" class="data row4 col5" >0.0411725</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col6" class="data row4 col6" >0.00273169</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col7" class="data row4 col7" >0.0482307</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow4_col8" class="data row4 col8" >-0.0450931</td> 
    </tr>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row5" class="row_heading level0 row5" >speechiness</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col0" class="data row5 col0" >0.072204</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col1" class="data row5 col1" >0.276206</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col2" class="data row5 col2" >-0.109983</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col3" class="data row5 col3" >-0.366762</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col4" class="data row5 col4" >0.0411725</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col5" class="data row5 col5" >1</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col6" class="data row5 col6" >0.00824055</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col7" class="data row5 col7" >-0.0269951</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow5_col8" class="data row5 col8" >0.149894</td> 
    </tr>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row6" class="row_heading level0 row6" >tempo</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col0" class="data row6 col0" >-0.0263097</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col1" class="data row6 col1" >-0.242089</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col2" class="data row6 col2" >0.195227</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col3" class="data row6 col3" >0.022215</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col4" class="data row6 col4" >0.00273169</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col5" class="data row6 col5" >0.00824055</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col6" class="data row6 col6" >1</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col7" class="data row6 col7" >-0.0253918</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow6_col8" class="data row6 col8" >0.0522212</td> 
    </tr>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row7" class="row_heading level0 row7" >track_id</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col0" class="data row7 col0" >-0.372282</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col1" class="data row7 col1" >0.0494541</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col2" class="data row7 col2" >0.140703</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col3" class="data row7 col3" >-0.275623</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col4" class="data row7 col4" >0.0482307</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col5" class="data row7 col5" >-0.0269951</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col6" class="data row7 col6" >-0.0253918</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col7" class="data row7 col7" >1</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow7_col8" class="data row7 col8" >0.0100698</td> 
    </tr>    <tr> 
        <th id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736alevel0_row8" class="row_heading level0 row8" >valence</th> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col0" class="data row8 col0" >-0.0138406</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col1" class="data row8 col1" >0.473165</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col2" class="data row8 col2" >0.0386027</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col3" class="data row8 col3" >-0.219967</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col4" class="data row8 col4" >-0.0450931</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col5" class="data row8 col5" >0.149894</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col6" class="data row8 col6" >0.0522212</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col7" class="data row8 col7" >0.0100698</td> 
        <td id="T_3a32beac_690f_11ef_9c5f_9af4e0a4736arow8_col8" class="data row8 col8" >1</td> 
    </tr></tbody> 
</table> 




```python
%%nose

def test_corr_matrix():
    assert all(corr_metrics == echonest_metrics.corr()) and isinstance(corr_metrics, pd.core.frame.DataFrame), \
        'The correlation matrix can be computed using the .corr() method.'
```






    1/1 tests passed




## 3. Splitting our data
<p>As mentioned earlier, it can be particularly useful to simplify our models and use as few features as necessary to achieve the best result. Since we didn't find any particularly strong correlations between our features, we can now split our data into an array containing our features, and another containing the labels - the genre of the track. </p>
<p>Once we have split the data into these arrays, we will perform some preprocessing steps to optimize our model development.</p>

#### Task 3: Instructions
Prepare our training and test sets and train our first classifier.

* Import the train_test_split() function from sklearn.model_selection module.
* Create features by storing all values of the echo_tracks DataFrame except for the "genre_top" and "track_id" columns.
* Create labels as an array of the "genre_top" column from the DataFrame.
* Split our projected data into train and tests, features and labels, respectively using train_test_split() with random_state=10.


```python
from sklearn.model_selection import train_test_split

# Create features by excluding 'genre_top' and 'track_id' columns
features = echo_tracks.drop(columns=['genre_top', 'track_id']).values

# Create labels from the 'genre_top' column
labels = echo_tracks['genre_top'].values

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels,  random_state=10
)


```


```python
%%nose

import sys

def test_features_labels():
    assert features.shape == (4802, 8), \
    """Did you drop "genre_top" from echo_tracks, and store all remaining values as features?"""
    assert labels.shape == (4802,), \
    """Did you store values from the "genre_top" column as labels?"""

def test_train_test_split_import():
    assert 'sklearn.model_selection' in list(sys.modules.keys()), \
    'Have you imported train_test_split from sklearn.model_selection?'
        
def test_train_test_split():
    train_test_res = train_test_split(features, labels, random_state=10)
    assert (train_features == train_test_res[0]).all(), \
    'Did you correctly call the train_test_split function?'
        
def test_correct_split():
    assert train_features.shape == (3601, 8), \
    """Did you correctly split the data? Expected a different shape for train_features."""
    assert test_features.shape == (1201, 8), \
    """Did you correctly split the data? Expected a different shape for test_features."""
    assert train_labels.shape == (3601,), \
    """Did you correctly split the data? Expected a different shape for train_labels."""
    assert test_labels.shape == (1201,), \
    """Did you correctly split the data? Expected a different shape for test_labels."""
```






    4/4 tests passed




## 4. Normalizing the feature data
<p>As mentioned earlier, it can be particularly useful to simplify our models and use as few features as necessary to achieve the best result. Since we didn't find any particular strong correlations between our features, we can instead use a common approach to reduce the number of features called <strong>principal component analysis (PCA)</strong>. </p>
<p>It is possible that the variance between genres can be explained by just a few features in the dataset. PCA rotates the data along the axis of highest variance, thus allowing us to determine the relative contribution of each feature of our data towards the variance between classes. </p>
<p>However, since PCA uses the absolute variance of a feature to rotate the data, a feature with a broader range of values will overpower and bias the algorithm relative to the other features. To avoid this, we must first normalize our train and test features. There are a few methods to do this, but a common way is through <em>standardization</em>, such that all features have a mean = 0 and standard deviation = 1 (the resultant is a z-score). </p>

#### Task 4: Instructions
Prepare our features and for training a model, and standardize the data.

* Import the StandardScaler from the sklearn.preprocessing module
* Define an instance of the StandardScaler called scaler without passing any arguments
* Use the scaler variable's fit_transform method to scale train_features and save to a new variable called scaled_train_features.
* Scale the test features using the scaler's transform method, passing test_features.


```python
from sklearn.preprocessing import StandardScaler

# Create an instance of StandardScaler
scaler = StandardScaler()

# Fit and transform the training features
scaled_train_features = scaler.fit_transform(train_features)

# Transform the test features using the same scaler
scaled_test_features = scaler.transform(test_features)

# Check the shape and basic statistics of the scaled data to ensure it's processed correctly
print("Scaled Training Features Shape:", scaled_train_features.shape)
print("Scaled Test Features Shape:", scaled_test_features.shape)

```

    Scaled Training Features Shape: (3601, 8)
    Scaled Test Features Shape: (1201, 8)



```python
%%nose

import sys
import numpy as np

# def test_labels_df():
#     try:
#         pd.testing.assert_series_equal(labels, echo_tracks['genre_top'])
#     except AssertionError:
#         assert False, 'Does your labels DataFrame only contain the genre_top column?'
        
def test_standardscaler_import():
    assert 'sklearn.preprocessing' in list(sys.modules.keys()), \
    'The StandardScaler can be imported from sklearn.preprocessing.'
        
def test_scaled_features():
    assert scaled_train_features[0].tolist() == [-1.3189452160155823,
 -1.748936113215404,
 0.5183796247907855,
 -0.2981419458739739,
 -0.19909374640763283,
 -0.41175479316875396,
 -0.911269482360871,
 -0.3436413082337475], \
    "Use the StandardScaler's fit_transform method on train_features."
    assert scaled_test_features[0].tolist() == [-1.3182917030552226,
 -1.6238218896488739,
 1.3841707828629735,
 -1.3119421397560926,
 2.1929908647262364,
 0.03499652489786962,
 1.9228785168921492,
 -0.2813786091336706], \
    "Use the StandardScaler's transform method on test_features."
```






    2/2 tests passed




## 5. Principal Component Analysis on our scaled data
<p>Now that we have preprocessed our data, we are ready to use PCA to determine by how much we can reduce the dimensionality of our data. We can use <strong>scree-plots</strong> and <strong>cumulative explained ratio plots</strong> to find the number of components to use in further analyses.</p>
<p>Scree-plots display the number of components against the variance explained by each component, sorted in descending order of variance. Scree-plots help us get a better sense of which components explain a sufficient amount of variance in our data. When using scree plots, an 'elbow' (a steep drop from one data point to the next) in the plot is typically used to decide on an appropriate cutoff.</p>

#### Task 5: Instructions
Use PCA to determine the explained variance of our features.

* Import the matplotlib.pyplot module as plt, and our PCA() class from sklearn.decomposition
* Create our PCA class using PCA(), fit the model on our scaled_train_features using PCA.fit(), and retrieve the explained variance ratio
* Make a scree plot of the variance explained by each component

We run PCA on all our features at first, which is done by default if n_components is not specified.


```python
%matplotlib inline

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create an instance of PCA
pca = PCA()

# Fit PCA on the scaled training features
pca.fit(scaled_train_features)

# Get the explained variance ratios
exp_variance = pca.explained_variance_ratio_

# Plot the explained variance using a barplot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')


```




    Text(0.5,0,'Principal Component #')




![png](output_18_1.png)



```python
%%nose

import sklearn
import numpy as np
import sys

def test_pca_import():
    assert ('sklearn.decomposition' in list(sys.modules.keys())), \
    'Have you imported the PCA object from sklearn.decomposition?'

def test_pca_obj():
    assert isinstance(pca, sklearn.decomposition.PCA), \
    "Use scikit-learn's PCA() object to create your own PCA object here."
        
def test_exp_variance():
    rounded_array = exp_variance
    rounder = lambda t: round(t, ndigits = 2)
    vectorized_round = np.vectorize(rounder)
    assert (vectorized_round(exp_variance)).all() == np.array([0.24, 0.18, 0.14, 0.13, 0.11, 0.09, 0.07, 0.05]).all(), \
    'Following the PCA fit, the explained variance ratios can be obtained via the explained_variance_ratio_ method.'
        
def test_scree_plot():
    expected_xticks = [float(n) for n in list(range(-1, 9))]
    assert list(ax.get_xticks()) == expected_xticks, \
    'Plot the number of pca components (on the x-axis) against the explained variance (on the y-axis).'
```






    4/4 tests passed




## 6. Further visualization of PCA
<p>Unfortunately, there does not appear to be a clear elbow in this scree plot, which means it is not straightforward to find the number of intrinsic dimensions using this method. </p>
<p>But all is not lost! Instead, we can also look at the <strong>cumulative explained variance plot</strong> to determine how many features are required to explain, say, about 85% of the variance (cutoffs are somewhat arbitrary here, and usually decided upon by 'rules of thumb'). Once we determine the appropriate number of components, we can perform PCA with that many components, ideally reducing the dimensionality of our data.</p>

#### Task 6: Instructions
Plot the cumulative explained variance of our PCA.

* Import numpy as np.
* Calculate the cumulative sums of our explained variance using np.cumsum().
* Plot the cumulative explained variances using ax.plot and look for the number of components at which we can account for >85% of our variance.


```python
# Import numpy
import numpy as np

# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# # Plot the cumulative explained variance and draw a dashed line at 0.85.
# fig, ax = plt.subplots()
# ax.plot(cum_exp_variance, marker='o', linestyle='-', color='b')
# ax.axhline(y=0.85, linestyle='--')


# Plot the cumulative explained variance
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cum_exp_variance, marker='o', linestyle='-', color='b')
ax.axhline(y=0.85, linestyle='--', color='r', label='85% Variance Threshold')

# Label the axes and add a title
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('Cumulative Explained Variance vs. Number of Principal Components')
ax.legend()

# Show the plot
plt.show()

```


![png](output_22_0.png)



```python
%%nose

import sys

def test_np_import():
    assert 'numpy' in list(sys.modules.keys()), \
    'Have you imported numpy?'

def test_cumsum():
    cum_exp_variance_correct = np.cumsum(exp_variance)
    assert all(cum_exp_variance == cum_exp_variance_correct), \
    'Use np.cumsum to calculate the cumulative sum of the exp_variance array.'
    
# def test_n_comp():
#     assert n_components == 5, \
#     ('Check the values in cum_exp_variance if it is difficult '
#     'to determine the number of components from the plot.')
    
    
# def test_trans_pca():
#     pca_test = PCA(n_components, random_state=10)
#     pca_test.fit(scaled_train_features)
#     assert (pca_projection == pca_test.transform(scaled_train_features)).all(), \
#     'Transform the scaled features and assign them to the pca_projection variable.'
```






    2/2 tests passed




## 7. Projecting on to our features
<p>We saw from the plot that 6 features (remember indexing starts at 0) can explain 85% of the variance! </p>
<p>Therefore, we can use 6 components to perform PCA and reduce the dimensionality of our train and test features.</p>

#### Task 7: Instructions
Perform Principal Component Analysis.

* Create pca by calling PCA, setting n_components equal to 6 and random_state=10.
* Use the pca variable's fit_transform method on scaled_train_features, saving as train_pca.
* Use the pca variable's transform method on scaled_test_features, saving as test_pca.


```python
# Create PCA instance with n_components=6 and random_state=10
pca = PCA(n_components=6, random_state=10)

# Fit and transform the scaled training features using PCA
train_pca = pca.fit_transform(scaled_train_features)

# Transform the scaled test features using the same PCA instance
test_pca = pca.transform(scaled_test_features)

# Optionally, print the shapes of the transformed data to verify
print("Training PCA Shape:", train_pca.shape)
print("Testing PCA Shape:", test_pca.shape)

```

    Training PCA Shape: (3601, 6)
    Testing PCA Shape: (1201, 6)



```python
%%nose

import sys
import sklearn

def test_pca_import():
    assert ('sklearn.decomposition' in list(sys.modules.keys())), \
    'Have you imported the PCA object from sklearn.decomposition?'
    
def test_pca_obj():
    assert isinstance(pca, sklearn.decomposition.PCA), \
    "Use scikit-learn's PCA() object to create your own PCA object here."    
    
def test_trans_pca():
    pca_copy = PCA(n_components=6, random_state=10)
    pca_copy.fit(scaled_train_features)
    assert train_pca.all() == pca_copy.transform(scaled_train_features).all(), \
    'Fit and transform the scaled training features and assign them to the train_pca variable.'
    pca_test = pca_copy.transform(scaled_test_features)
    assert test_pca.all() == pca_copy.transform(scaled_test_features).all()
```






    3/3 tests passed




## 8. Train a decision tree to classify genre
<p>Now we can use the lower dimensional PCA projection of the data to classify songs into genres. </p>
<p>Here, we will be using a simple algorithm known as a decision tree. Decision trees are rule-based classifiers that take in features and follow a 'tree structure' of binary decisions to ultimately classify a data point into one of two or more categories. In addition to being easy to both use and interpret, decision trees allow us to visualize the 'logic flowchart' that the model generates from the training data.</p>
<p>Here is an example of a decision tree that demonstrates the process by which an input image (in this case, of a shape) might be classified based on the number of sides it has and whether it is rotated.</p>
<p><img src="https://assets.datacamp.com/production/project_449/img/simple_decision_tree.png" alt="Decision Tree Flow Chart Example" width="350px"></p>

#### Task 8: Instructions
Train our first classifier.

* Import the DecisionTreeClassifier from sklearn.tree module.
* Create our decision tree classifier using DecisionTreeClassifier() and random_state=10.
* Train the model by passing train_pca and train_labels to the .fit() method.
* Find the predicted labels of the test_pca from our trained model using the model.predict() notation.


```python
# Import Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create our decision tree classifier with a random_state for reproducibility
tree = DecisionTreeClassifier(random_state=10)

# Train the decision tree classifier on the PCA-transformed training features and labels
tree.fit(train_pca, train_labels)

# Predict the labels for the test data using the trained model
pred_labels_tree = tree.predict(test_pca)

# Optionally, print some of the predicted labels to verify
print("Predicted Labels for Test Data (first 10):", pred_labels_tree[:10])

```

    Predicted Labels for Test Data (first 10): ['Rock' 'Rock' 'Rock' 'Rock' 'Hip-Hop' 'Rock' 'Rock' 'Rock' 'Rock' 'Rock']



```python
%%nose

import sys

# def test_train_test_split_import():
#     assert 'sklearn.model_selection' in list(sys.modules.keys()), \
#         'Have you imported train_test_split from sklearn.model_selection?'

    
def test_decision_tree_import():
    assert 'sklearn.tree' in list(sys.modules.keys()), \
    'Have you imported DecisionTreeClassifier from sklearn.tree?'
    
    
# def test_train_test_split():
#     train_test_res = train_test_split(pca_projection, labels, random_state=10)
#     assert (train_features == train_test_res[0]).all(), \
#         'Did you correctly call the train_test_split function?'
    
    
def test_tree():
    assert tree.get_params() == DecisionTreeClassifier(random_state=10).get_params(), \
    'Did you create the decision tree correctly?'
    
    
def test_tree_fit():
    assert hasattr(tree, 'classes_'), \
    'Did you fit the tree to the training data?'
    
    
def test_tree_pred():
    assert (pred_labels_tree == 'Rock').sum() == 971, \
    'Did you correctly use the fitted tree object to make a prediction from test_pca?'
```






    4/4 tests passed




## 9. Compare our decision tree to a logistic regression
<p>Although our tree's performance is decent, it's a bad idea to immediately assume that it's therefore the perfect tool for this job -- there's always the possibility of other models that will perform even better! It's always a worthwhile idea to at least test a few other algorithms and find the one that's best for our data.</p>
<p>Sometimes simplest is best, and so we will start by applying <strong>logistic regression</strong>. Logistic regression makes use of what's called the logistic function to calculate the odds that a given data point belongs to a given class. Once we have both models, we can compare them on a few performance metrics, such as false positive and false negative rate (or how many points are inaccurately classified). </p>

#### Task 9: Instructions
Train our logistic regression and compare the performance with our decision tree.

* Create our logistic regression model using LogisticRegression() and set random_state to 10.
* Train the model using the model.fit() notation and assign the predicted labels for the test_features to pred_labels_logit.
* Import the classification_report from the sklearn.metrics package
* Print the classification reports for our trained Decision Tree and Logistic Regression models.


```python
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Create our logistic regression model with a random_state for reproducibility
logreg = LogisticRegression(random_state=10)

# Train the logistic regression model on the PCA-transformed training features and labels
logreg.fit(train_pca, train_labels)

# Predict the labels for the test data using the trained logistic regression model
pred_labels_logit = logreg.predict(test_pca)

# Generate classification reports for both models
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

# Print the classification reports
print("Decision Tree:\n", class_rep_tree)
print("Logistic Regression:\n", class_rep_log)

```

    Decision Tree:
                  precision    recall  f1-score   support
    
        Hip-Hop       0.70      0.70      0.70       229
           Rock       0.93      0.93      0.93       972
    
    avg / total       0.88      0.88      0.88      1201
    
    Logistic Regression:
                  precision    recall  f1-score   support
    
        Hip-Hop       0.76      0.57      0.65       229
           Rock       0.90      0.96      0.93       972
    
    avg / total       0.88      0.88      0.88      1201
    



```python
%%nose

def test_logreg():
    assert logreg.get_params() == LogisticRegression(random_state=10).get_params(), \
    'The logreg variable should be created using LogisticRegression().'

    
def test_logreg_pred():
    assert abs((pred_labels_logit == 'Rock').sum() - 1028) < 7, \
    'The labels should be predicted from the test_features.'
    
    
def test_class_rep_tree():
    assert isinstance(class_rep_tree, str), \
    'Did you create the classification report correctly for the decision tree?'
    
    
def test_class_rep_log():
    assert isinstance(class_rep_log, str), \
    'Did you create the classification report correctly for the logistic regression?'
```






    4/4 tests passed




## 10. Balance our data for greater performance
<p>Both our models do similarly well, boasting an average precision of 87% each. However, looking at our classification report, we can see that rock songs are fairly well classified, but hip-hop songs are disproportionately misclassified as rock songs. </p>
<p>Why might this be the case? Well, just by looking at the number of data points we have for each class, we see that we have far more data points for the rock classification than for hip-hop, potentially skewing our model's ability to distinguish between classes. This also tells us that most of our model's accuracy is driven by its ability to classify just rock songs, which is less than ideal.</p>
<p>To account for this, we can weight the value of a correct classification in each class inversely to the occurrence of data points for each class. Since a correct classification for "Rock" is not more important than a correct classification for "Hip-Hop" (and vice versa), we only need to account for differences in <em>sample size</em> of our data points when weighting our classes here, and not relative importance of each class. </p>

#### Task 10: Instructions
Balance our dataset such that the number of tracks for each genre is the same.

* Subset only the hip-hop tracks from echo_tracks using df.loc[], and the same for the rock tracks
* Sample rock_only such that there is the same number of data points as there are hip-hop data points. Set the random_state to 10.
* Concatenate the rock_only and hop_only (in that order) DataFrames using the pd.concat() function by passing a list of these DataFrames.
* Redefine our train and test sets using train_test_split with features and labels created from the balanced dataframe.


```python
# Subset only the hip-hop tracks
hop_only = echo_tracks[echo_tracks['genre_top'] == 'Hip-Hop']

# Subset only the rock tracks
rock_only = echo_tracks[echo_tracks['genre_top'] == 'Rock']

# Sample rock_only to have the same number of data points as hop_only
rock_only = rock_only.sample(n=len(hop_only), random_state=10)

# Concatenate the rock_only and hop_only DataFrames
rock_hop_bal = pd.concat([rock_only, hop_only], axis=0)

# Define features and labels from the balanced DataFrame
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1)
labels = rock_hop_bal['genre_top']

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, random_state=10
)

# Standardize the features
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

# Perform PCA with 6 components
pca = PCA(n_components=6, random_state=10)
train_pca = pca.fit_transform(scaled_train_features)
test_pca = pca.transform(scaled_test_features)


```


```python
%%nose

def test_hop_only():
    try:
        pd.testing.assert_frame_equal(hop_only, echo_tracks.loc[echo_tracks['genre_top'] == 'Hip-Hop'])
    except AssertionError:
        assert False, "The hop_only data frame was not assigned correctly."
        

def test_rock_only():
    try:
        pd.testing.assert_frame_equal(
            rock_only, echo_tracks.loc[echo_tracks['genre_top'] == 'Rock'].sample(hop_only.shape[0], random_state=10))
    except AssertionError:
        assert False, "The rock_only data frame was not assigned correctly."
        
        
def test_rock_hop_bal():
    hop_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Hip-Hop']
    rock_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Rock'].sample(hop_only.shape[0], random_state=10)
    try:
        pd.testing.assert_frame_equal(
            rock_hop_bal, pd.concat([rock_only, hop_only]))
    except AssertionError:
        assert False, "The rock_hop_bal data frame was not assigned correctly."
        
        
def test_train_features():
    assert round(train_pca[0][0], 4) == -0.6434 and round(test_pca[0][0], 4) == 0.4368, \
    'The train_test_split was not performed correctly.'
```






    4/4 tests passed




## 11. Does balancing our dataset improve model bias?
<p>We've now balanced our dataset, but in doing so, we've removed a lot of data points that might have been crucial to training our models. Let's test to see if balancing our data improves model bias towards the "Rock" classification while retaining overall classification performance. </p>
<p>Note that we have already reduced the size of our dataset and will go forward without applying any dimensionality reduction. In practice, we would consider dimensionality reduction more rigorously when dealing with vastly large datasets and when computation times become prohibitively large.</p>

#### Task 11: Instructions
Compare the two model performances on the balanced data.

* Create and train your decision tree using DecisionTreeClassifier() and a random state of 10, then predict on the test_features.
* Create and train your logistic regression using LogisticRegression() and a random state of 10, then predict on the test_features.
* Compare the performance of the two models using classification_report().


```python
# Train the Decision Tree classifier on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels)
pred_labels_tree = tree.predict(test_pca)

# Train the Logistic Regression classifier on the balanced data
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca, train_labels)
pred_labels_logit = logreg.predict(test_pca)

# Generate and print classification reports for both models
print("Decision Tree:\n", classification_report(test_labels, pred_labels_tree))
print("Logistic Regression:\n", classification_report(test_labels, pred_labels_logit))

```

    Decision Tree:
                  precision    recall  f1-score   support
    
        Hip-Hop       0.75      0.79      0.77       230
           Rock       0.77      0.73      0.75       225
    
    avg / total       0.76      0.76      0.76       455
    
    Logistic Regression:
                  precision    recall  f1-score   support
    
        Hip-Hop       0.81      0.83      0.82       230
           Rock       0.83      0.80      0.82       225
    
    avg / total       0.82      0.82      0.82       455
    



```python
%%nose

def test_tree_bal():
    assert (pred_labels_tree == 'Rock').sum() == 213, \
    'The pred_labels_tree variable should contain the predicted labels from the test_features.'
    
    
def test_logit_bal():
    assert (pred_labels_logit == 'Rock').sum() == 219, \
    'The pred_labels_logit variable should contain the predicted labels from the test_features.'
```






    2/2 tests passed




## 12. Using cross-validation to evaluate our models
<p>Success! Balancing our data has removed bias towards the more prevalent class. To get a good sense of how well our models are actually performing, we can apply what's called <strong>cross-validation</strong> (CV). This step allows us to compare models in a more rigorous fashion.</p>
<p>Before we can perform cross-validation we will need to create pipelines to scale our data, perform PCA, and instantiate our model of choice - <code>DecisionTreeClassifier</code> or <code>LogisticRegression</code>.</p>
<p>Since the way our data is split into train and test sets can impact model performance, CV attempts to split the data multiple ways and test the model on each of the splits. Although there are many different CV methods, all with their own advantages and disadvantages, we will use what's known as <strong>K-fold</strong> CV here. K-fold first splits the data into K different, equally sized subsets. Then, it iteratively uses each subset as a test set while using the remainder of the data as train sets. Finally, we can then aggregate the results from each fold for a final model performance score.</p>

#### Task 12: Instructions
Use cross-validation to get a better sense of model performance.

* Create a variable called kf to store your cv using KFold() with 10 folds.
* Train each of your pipelines using cross_val_score() on the original features and labels variables, setting cv equal to kf.
* Print the mean of the cross-validation scores for each pipeline using np.mean().
* 


```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

# Create pipelines for Decision Tree and Logistic Regression
tree_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=6)),
    ("tree", DecisionTreeClassifier(random_state=10))
])

logreg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=6)),
    ("logreg", LogisticRegression(random_state=10))
])

# Set up K-Fold cross-validation with 10 folds
kf = KFold(n_splits=10)


# Train our models using KFold cv
# Perform cross-validation for Decision Tree
tree_score = cross_val_score(tree_pipe, features, labels, cv=kf)

# Perform cross-validation for Logistic Regression
logit_score = cross_val_score(logreg_pipe, features, labels, cv=kf)

# Print the mean of each array of scores
print("Decision Tree:", np.mean(tree_score))
print("Logistic Regression:", np.mean(logit_score))

```

    Decision Tree: 0.7219780219780221
    Logistic Regression: 0.773076923076923



```python
%%nose

def test_kf():
    assert kf.__repr__() == 'KFold(n_splits=10, random_state=None, shuffle=False)', \
    'The k-fold cross-validation was not setup correctly.'
    
    
def test_tree_score():
    assert np.isclose(round((tree_score.sum() / tree_score.shape[0]), 4), 0.722, atol=1e-3), \
    'The tree_score was not calculated correctly.'
    
    
def test_log_score():
    assert np.isclose(round((logit_score.sum() / logit_score.shape[0]), 4), 0.7731, atol=1e-3), \
    'The logit_score was not calculated correctly.'
```






    3/3 tests passed



