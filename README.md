# Coursera Course Analysis

## Data preparation

Data downloaded from [Kaggle](https://www.kaggle.com/datasets/siddharthm1698/coursera-course-dataset) website as `coursea_data.csv` file and saved in local folder. For data analysis and visualization we need to set up the environment. First of all we need to import necessary libraries to the notebook and provide special function in the notebook - `%matplotlib inline` With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.

Importing libraries for working with data.


```python
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm

```

Set up seaborn styles


```python
# Set up custom warm palette
custom_palette = sns.color_palette("Blues_r")
sns.set_palette(custom_palette)

# Set up whitegrid
sns.set(style="whitegrid")
```

Now we can import our data to notebook, read it from .csv and convert it to the data frame. Base by dataset provider we know we already have number to identify dataset row, so we can set this column as index number for our rows in data. Setting up data path and loading it to data frame.


```python
data_path = "data/coursea_data.csv"
df = pd.read_csv(data_path, index_col=0)
```

First check the shape of the data.


```python
df.shape
```




    (891, 6)



From brief view we can see data loaded successfully. It has 891 rows (1 row added for column names) and 6 columns of data with additional 1 column for rows index. Now check what kind of data we have, we can see first few rows of the data, and get the data types of it.


```python
df.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>course_title</th>
      <th>course_organization</th>
      <th>course_Certificate_type</th>
      <th>course_rating</th>
      <th>course_difficulty</th>
      <th>course_students_enrolled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>134</th>
      <td>(ISC)² Systems Security Certified Practitioner...</td>
      <td>(ISC)²</td>
      <td>SPECIALIZATION</td>
      <td>4.7</td>
      <td>Beginner</td>
      <td>5.3k</td>
    </tr>
    <tr>
      <th>743</th>
      <td>A Crash Course in Causality:  Inferring Causal...</td>
      <td>University of Pennsylvania</td>
      <td>COURSE</td>
      <td>4.7</td>
      <td>Intermediate</td>
      <td>17k</td>
    </tr>
    <tr>
      <th>874</th>
      <td>A Crash Course in Data Science</td>
      <td>Johns Hopkins University</td>
      <td>COURSE</td>
      <td>4.5</td>
      <td>Mixed</td>
      <td>130k</td>
    </tr>
    <tr>
      <th>413</th>
      <td>A Law Student's Toolkit</td>
      <td>Yale University</td>
      <td>COURSE</td>
      <td>4.7</td>
      <td>Mixed</td>
      <td>91k</td>
    </tr>
    <tr>
      <th>635</th>
      <td>A Life of Happiness and Fulfillment</td>
      <td>Indian School of Business</td>
      <td>COURSE</td>
      <td>4.8</td>
      <td>Mixed</td>
      <td>320k</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    course_title                 object
    course_organization          object
    course_Certificate_type      object
    course_rating               float64
    course_difficulty            object
    course_students_enrolled     object
    dtype: object



Base by brief view we can see 7 data columns (column 1 is additional index column):
<ol>
    <li># - Provide number to identify dataset. <strong>Data type: object</strong></li>
    <li>course_title - Provide the course title. <strong>Data type: object</strong></li>
    <li>course_organization - Provide organization which conducting the courses.<strong>Data type: object</strong></li>
    <li>course_Certificate_type - Provide details what certification available in courses.<strong>Data type: object</strong></li>
    <li>course_rating - Provide ratings for given course.<strong>Data type: float64</strong></li>
    <li>course_difficulty - Provide information how difficult or what level is the course.<strong>Data type: object</strong></li>
    <li>course_students_enrolled - Provide the number of students that are enrolled in the course.<strong>Data type: object</strong></li>
</ol>

## Data cleaning

Before working with data we need to make some adjustments to it, we need to check if there are no missing values, also if there are any duplicate values witch can be removed if necessary, check if column data types are valid for that column.

We can see a problem in column: `course_students_enrolled` it is object type holding numeric and string values, as we do some calculations with numeric values, data type need to be changed, we can fix it by adding new column with converted values in it. For these calculations we create function `convert_with_k` witch check if row value have "k" in the string and multiplies it by 1,000, if value have "m" multiply it by 1,000,000 otherwise just converts it to numeric value. Function provide some security check in column, it will fill "NaN" value to the row if it can't convert it normally.


```python
def convert_with_k(value):
    if "k" in str(value):
        return pd.to_numeric(value.replace("k", "")) * 1000
    elif "m" in str(value):
        return pd.to_numeric(value.replace("m", "")) * 1000000
    else:
        return pd.to_numeric(value, errors="coerce") # if errors occurs by converting value, return NaN
```


```python
df["students_count"] = df["course_students_enrolled"].apply(convert_with_k).astype(int)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>course_title</th>
      <th>course_organization</th>
      <th>course_Certificate_type</th>
      <th>course_rating</th>
      <th>course_difficulty</th>
      <th>course_students_enrolled</th>
      <th>students_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>134</th>
      <td>(ISC)² Systems Security Certified Practitioner...</td>
      <td>(ISC)²</td>
      <td>SPECIALIZATION</td>
      <td>4.7</td>
      <td>Beginner</td>
      <td>5.3k</td>
      <td>5300</td>
    </tr>
    <tr>
      <th>743</th>
      <td>A Crash Course in Causality:  Inferring Causal...</td>
      <td>University of Pennsylvania</td>
      <td>COURSE</td>
      <td>4.7</td>
      <td>Intermediate</td>
      <td>17k</td>
      <td>17000</td>
    </tr>
    <tr>
      <th>874</th>
      <td>A Crash Course in Data Science</td>
      <td>Johns Hopkins University</td>
      <td>COURSE</td>
      <td>4.5</td>
      <td>Mixed</td>
      <td>130k</td>
      <td>130000</td>
    </tr>
    <tr>
      <th>413</th>
      <td>A Law Student's Toolkit</td>
      <td>Yale University</td>
      <td>COURSE</td>
      <td>4.7</td>
      <td>Mixed</td>
      <td>91k</td>
      <td>91000</td>
    </tr>
    <tr>
      <th>635</th>
      <td>A Life of Happiness and Fulfillment</td>
      <td>Indian School of Business</td>
      <td>COURSE</td>
      <td>4.8</td>
      <td>Mixed</td>
      <td>320k</td>
      <td>320000</td>
    </tr>
  </tbody>
</table>
</div>



As we can see new column `students_count` was created with converted values in it. Now let's use `isnull` function to check if no NaN values appeared in new column. If function return `True` we need to do additional configuration in function to catch all valid values, if return `False` we can assume all data converted successfully.


```python
df.isnull().any()
```




    course_title                False
    course_organization         False
    course_Certificate_type     False
    course_rating               False
    course_difficulty           False
    course_students_enrolled    False
    students_count              False
    dtype: bool



Function return `False`, lastly we need to check for any duplicate values.


```python
df.duplicated().any()
```




    False



Function returned `False`, now we know there are no duplicates in dataset, we can check data for possible outliers.


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>course_rating</th>
      <th>students_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>8.910000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.677329</td>
      <td>9.055208e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.162225</td>
      <td>1.819365e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.300000</td>
      <td>1.500000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.600000</td>
      <td>1.750000e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.700000</td>
      <td>4.200000e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.800000</td>
      <td>9.950000e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>3.200000e+06</td>
    </tr>
  </tbody>
</table>
</div>



As we can see course_rating looks good, number distribution is not very big and good for analysis, but students_count looks strange, but we know that it show how many students enrolled to course, some course can have few thousands while others can have over few millions, so we consider it as outlier, but we will still explore it as valid data without modifying it.

## Exploratory data analysis

Through exploratory data analysis, we aim to identify the most popular course and predict the course that students are most likely to choose.

Now we can start exploring the data and make some analysis of it. First let's check how many different data we have in our dataset, for this we can use `nunique` function to get unique values in every column.


```python
df.nunique()
```




    course_title                888
    course_organization         154
    course_Certificate_type       3
    course_rating                14
    course_difficulty             4
    course_students_enrolled    205
    students_count              205
    dtype: int64



Base by unique values some question occurs. Why course title have 888 different values, as we know there are 890 rows, and we check for duplicates and get nothing, so that mean we have some courses with same title but different features, let's check those courses.


```python
df[df["course_title"].duplicated(keep="first") | df["course_title"].duplicated(keep=False)]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>course_title</th>
      <th>course_organization</th>
      <th>course_Certificate_type</th>
      <th>course_rating</th>
      <th>course_difficulty</th>
      <th>course_students_enrolled</th>
      <th>students_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>756</th>
      <td>Developing Your Musicianship</td>
      <td>Berklee College of Music</td>
      <td>COURSE</td>
      <td>4.8</td>
      <td>Mixed</td>
      <td>41k</td>
      <td>41000</td>
    </tr>
    <tr>
      <th>205</th>
      <td>Developing Your Musicianship</td>
      <td>Berklee College of Music</td>
      <td>SPECIALIZATION</td>
      <td>4.8</td>
      <td>Beginner</td>
      <td>54k</td>
      <td>54000</td>
    </tr>
    <tr>
      <th>181</th>
      <td>Machine Learning</td>
      <td>University of Washington</td>
      <td>SPECIALIZATION</td>
      <td>4.6</td>
      <td>Intermediate</td>
      <td>290k</td>
      <td>290000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Machine Learning</td>
      <td>Stanford University</td>
      <td>COURSE</td>
      <td>4.9</td>
      <td>Mixed</td>
      <td>3.2m</td>
      <td>3200000</td>
    </tr>
    <tr>
      <th>241</th>
      <td>Marketing Digital</td>
      <td>Universidade de São Paulo</td>
      <td>COURSE</td>
      <td>4.8</td>
      <td>Beginner</td>
      <td>81k</td>
      <td>81000</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Marketing Digital</td>
      <td>Universidad Austral</td>
      <td>SPECIALIZATION</td>
      <td>4.7</td>
      <td>Beginner</td>
      <td>39k</td>
      <td>39000</td>
    </tr>
  </tbody>
</table>
</div>



As we assume we have some courses with same title, but as we see courses have different organization, certification type and course difficulty. Other thing we know that some features in our dataset have hundreds of values while others have only few values, like `course_Certificate_type` and `course_difficulty`. For features with low number of unique values we can explore the actual unique values to understand the categories using `unique`.


```python
df["course_Certificate_type"].unique()
```




    array(['SPECIALIZATION', 'COURSE', 'PROFESSIONAL CERTIFICATE'],
          dtype=object)



`course_Certificate_type` - have 3 different types: **SPECIALIZATION**, **COURSE** and **PROFESSIONAL CERTIFICATE**. For this feature lets check the distribution of certificate types.


```python
plt.figure(figsize=(10,6))
plt.title("Distribution Of Certificate types")
plt.xlabel("Certificate types")
plt.ylabel("Count")
sns.histplot(data=df, x="course_Certificate_type", palette=custom_palette)
```

<p align="center">
  <img src="Images/output_35_1.png">
</p>
    


Plot showing that "SPECIALIZATION" and "COURSE" types are dominating over "PROFESSIONAL CERTIFICATE"                            


```python
df["course_difficulty"].unique()
```




    array(['Beginner', 'Intermediate', 'Mixed', 'Advanced'], dtype=object)



`course_difficulty` - have 4 different types: **Beginner**, **Intermediate**, **Mixed** and **Advanced**. For this feature we will plot distribution of difficulty.


```python
plt.figure(figsize=(10,6))
plt.title("Distribution Of Course Difficulty")
plt.xlabel("Course Difficulty")
plt.ylabel("Count")
sns.histplot(data=df, x="course_difficulty")
```    
<p align="center">
  <img src="Images/output_39_1.png">
</p>
    


This distribution show that "Beginner" course are most popular over all, while "Intermediate" and "Mixed" are both almost the same, and only little piece are for "Advanced" course.

Base by these values we see some groups in our data. We can try to analyse it more deeply. But first we can use `describe` function to get some statistical view of numerical features in our dataset, so we can use combination of different features to find more insigns.


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>course_rating</th>
      <th>students_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>8.910000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.677329</td>
      <td>9.055208e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.162225</td>
      <td>1.819365e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.300000</td>
      <td>1.500000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.600000</td>
      <td>1.750000e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.700000</td>
      <td>4.200000e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.800000</td>
      <td>9.950000e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>3.200000e+06</td>
    </tr>
  </tbody>
</table>
</div>



We can see some information about course ratings and student count. Student count goes from 1.5 thousand to 3.2 million, course rating goes from minimum 3.3 to maximum 5.0, while average of ratings are 4.7, we can plot the course ratings to check the distribution of it.


```python
plt.figure(figsize=(10,6))
plt.title("Distribution Of Course Ratings")
plt.xlabel("Course Rating")
plt.ylabel("Frequency")
sns.kdeplot(data=df, x="course_rating")
```

<p align="center">
  <img src="Images/output_44_1.png">
</p>


We can see that courses have high rating, most of them goes from 4.1 to 5.0, people tend to rate course higher than lower. Now let's check distribution of students based on enrollment size.


```python
# Splitting data
students_more = df[df["students_count"] >= 1000000]["students_count"].sum()
students_less = df[df["students_count"] < 1000000]["students_count"].sum()

# Explode slice
explode = [0.1, 0]

# Custom pie colors
pie_colors = ["lightblue", "lightgray"]

# Pie plot
plt.figure(figsize=(8, 8))
plt.pie([students_more, students_less], 
        labels=["More than million", "Less than million"], 
        autopct='%1.1f%%', 
        startangle=180, 
        explode=explode, 
        colors=pie_colors)

# Set up title
plt.title("Distribution of Students Based on Enrollment Size")

```

<p align="center">
  <img src="Images/output_46_1.png">
</p>

From pie chart we find that about 10.5% of students are enrolled in course which have more than a million students in them while 89.5% of students enrolled in less popular courses. Let's find most popular courses.

### TOP 5 most popular courses

As wee know there are some courses with same title, to get the most popular course title we need to join same titles.


```python
# Group data by title and get top 5 values
grouped_title = df.groupby("course_title", as_index=False)["students_count"].sum()
grouped_title_top_five = grouped_title.sort_values(by=["students_count"], ascending=False).head(5)

# Plot barplot
plt.figure(figsize=(10, 6)) # Size
sns.barplot(data=grouped_title_top_five, x="course_title", y="students_count", palette=custom_palette)

# Set up labels
plt.title("Top 5 Most Popular Courses") # Title
plt.xticks(rotation=45, ha="right") # Rotate x Labels by 45 degrees
plt.xlabel("Course Title") # x Label
plt.ylabel("Enrolled Students (Million)") # y Label
```

<p align="center">
  <img src="Images/output_50_1.png">
</p>



```python
# Get exact number of students in courses
grouped_title_top_five.rename(columns={'course_title': 'Course Title', 'students_count': 'Total Students'})
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Course Title</th>
      <th>Total Students</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>562</th>
      <td>Machine Learning</td>
      <td>3490000</td>
    </tr>
    <tr>
      <th>812</th>
      <td>The Science of Well-Being</td>
      <td>2500000</td>
    </tr>
    <tr>
      <th>685</th>
      <td>Python for Everybody</td>
      <td>1500000</td>
    </tr>
    <tr>
      <th>671</th>
      <td>Programming for Everybody (Getting Started wit...</td>
      <td>1300000</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Data Science</td>
      <td>830000</td>
    </tr>
  </tbody>
</table>
</div>

It appears that **Machine Learning** is top course that students pick. It has almost 3.5 million students enrolled to the course. Second place with 2.5 million students is **The Science of Well-Being**, almost 1 million lower than first place. Third and forth places takes **Python for Everybody** and **Programming for Everybody (Getting Started with Python) difference between each other is 200 thousand students. Last place with less than a million take "Data Science" course. It can be noted that 4 out of 5 places takes programming.


### TOP 5 organization that are the most popular to study in

To know which organization is most popular over all of them we need to join them and check popularity over students.


```python
# Group by organization, sort values and get top 5
grouped_by_cert = df.groupby("course_organization", as_index=False)["students_count"].sum()
grouped_by_cert_top = grouped_by_cert.sort_values(by=["students_count"], ascending=False).head(5)

# Plot barplot
plt.figure(figsize=(10, 6)) # Size
sns.barplot(data=grouped_by_cert_top, x="course_organization", y="students_count", palette=custom_palette)

# Set up title and labels
plt.title("Top 5 Most Popular Organizations") # Title
plt.xlabel("Organization") # x label
plt.xticks(rotation=45, ha="right") # Rotate 45 degrees
plt.ylabel("Enrolled Students (Million)") # y Label
```


<p align="center">
  <img src="Images/output_55_1.png">
</p>



```python
grouped_by_cert_top.rename(columns={"course_organization": "Course Organization", "students_count": "Students Count"})
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Course Organization</th>
      <th>Students Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135</th>
      <td>University of Michigan</td>
      <td>7437700</td>
    </tr>
    <tr>
      <th>138</th>
      <td>University of Pennsylvania</td>
      <td>5501300</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Stanford University</td>
      <td>4854000</td>
    </tr>
    <tr>
      <th>123</th>
      <td>University of California, Irvine</td>
      <td>4326000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Johns Hopkins University</td>
      <td>4298900</td>
    </tr>
  </tbody>
</table>
</div>

We find that first place of organization took by **University of Michigan** with more than 7 million students enrolled. In second place sitting **University of Pennsylvania** with 5.5 million. Third place is **Stanford University** with 4.8 million. For fourth place **University of California, Irvine** and fifth place **Johns Hopkins University** we see than both organizations are quite close with average 4.3 million students enrolled in each.

### TOP 1 most popular course based by course difficulty and certification type


```python
# Group data by title. type, difficulty and sum by student count
grouped_cert_diff = df.groupby(["course_title", 
                                "course_Certificate_type", 
                                "course_difficulty"], as_index=False)["students_count"].sum()

# Get top titleu sing idmax of student count
grouped_cert_diff_top = grouped_cert_diff.loc[grouped_cert_diff.groupby(["course_Certificate_type", 
                                                                          "course_difficulty"])["students_count"].idxmax()]
# Create a figure and 3 rows of subplots
fig, axes = plt.subplots(nrows=3, figsize=(10, 12))

# Plot Dim1
bar_plot1 = sns.barplot(
            data=grouped_cert_diff_top
            .where(grouped_cert_diff_top["course_Certificate_type"] == "SPECIALIZATION")
            .sort_values(by=["students_count"]), 
            x="students_count", 
            y="course_title", 
            hue="course_difficulty", 
            palette=custom_palette, ax=axes[0])

# Plot Dim2
bar_plot2 = sns.barplot(
            data=grouped_cert_diff_top
            .where(grouped_cert_diff_top["course_Certificate_type"] == "COURSE")
            .sort_values(by=["students_count"]), 
            x="students_count", 
            y="course_title", 
            hue="course_difficulty", 
            palette=custom_palette, ax=axes[1])

# Plot Dim3
bar_plot3 =sns.barplot(data=grouped_cert_diff_top.where(grouped_cert_diff_top["course_Certificate_type"] == "PROFESSIONAL CERTIFICATE").sort_values(by=["students_count"]), 
            x="students_count", 
            y="course_title", 
            hue="course_difficulty", 
            palette=custom_palette, ax=axes[2])

# Set title and labels
axes[0].set_title("Mots Popular Course By Type (SPECIALIZATION) And Difficulty")
axes[1].set_title("Mots Popular Course By Type (COURSE) And Difficulty")
axes[2].set_title("Mots Popular Course By Type (PROFESSIONAL CERTIFICATE) And Difficulty")

axes[0].set_xlabel("Enrolled Students (Million)")
axes[1].set_xlabel("Enrolled Students (Million)")
axes[2].set_xlabel("Enrolled Students")

axes[0].set_ylabel("Course Title")
axes[1].set_ylabel("Course Title")
axes[2].set_ylabel("Course Title")

legend1 = bar_plot1.legend(title="Course Difficulty", loc="upper right")
legend2 = bar_plot2.legend(title="Course Difficulty", loc="upper right")
legend3 = bar_plot3.legend(title="Course Difficulty", loc="upper right")

plt.subplots_adjust(hspace=0.5)
```


    
<p align="center">
  <img src="Images/output_59_0.png">
</p>



```python
grouped_cert_diff_top
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>course_title</th>
      <th>course_Certificate_type</th>
      <th>course_difficulty</th>
      <th>students_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>600</th>
      <td>Medical Neuroscience</td>
      <td>COURSE</td>
      <td>Advanced</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Financial Markets</td>
      <td>COURSE</td>
      <td>Beginner</td>
      <td>470000</td>
    </tr>
    <tr>
      <th>626</th>
      <td>Neural Networks and Deep Learning</td>
      <td>COURSE</td>
      <td>Intermediate</td>
      <td>630000</td>
    </tr>
    <tr>
      <th>563</th>
      <td>Machine Learning</td>
      <td>COURSE</td>
      <td>Mixed</td>
      <td>3200000</td>
    </tr>
    <tr>
      <th>420</th>
      <td>IBM Data Science</td>
      <td>PROFESSIONAL CERTIFICATE</td>
      <td>Beginner</td>
      <td>480000</td>
    </tr>
    <tr>
      <th>142</th>
      <td>Cloud Engineering with Google Cloud</td>
      <td>PROFESSIONAL CERTIFICATE</td>
      <td>Intermediate</td>
      <td>310000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Advanced Data Science with IBM</td>
      <td>SPECIALIZATION</td>
      <td>Advanced</td>
      <td>320000</td>
    </tr>
    <tr>
      <th>688</th>
      <td>Python for Everybody</td>
      <td>SPECIALIZATION</td>
      <td>Beginner</td>
      <td>1500000</td>
    </tr>
    <tr>
      <th>211</th>
      <td>Deep Learning</td>
      <td>SPECIALIZATION</td>
      <td>Intermediate</td>
      <td>690000</td>
    </tr>
  </tbody>
</table>
</div>

It can be noted the popularity distribution of certificate type and course difficulty and the top courses of different aspects. We can see **Machine learning** is the only one course with **Mixed** difficulty over all course types, also it has been observed that **Beginner** difficulty is most popular over course types. It appears students who take courses to be more beginner level.

### Best organizations based by certification type and rating


```python
# Group by organization and type also aggregate course_rating to mean and student_count to sum
grouped_data = df.groupby(["course_organization", 
                          "course_Certificate_type"], as_index=False).agg({"course_rating": "mean",
                                                                          "students_count": "sum"})
# Sort by course_rating and students_count
grouped_data_sorted = grouped_data.sort_values(["course_rating","students_count"], ascending=[False, False])

# Find all top organizations base by type, rating and student count
best_org_types = grouped_data_sorted.groupby("course_Certificate_type").head(1)
best_org_types
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>course_organization</th>
      <th>course_Certificate_type</th>
      <th>course_rating</th>
      <th>students_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>228</th>
      <td>Yonsei University</td>
      <td>COURSE</td>
      <td>4.9</td>
      <td>520000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Arizona State University</td>
      <td>PROFESSIONAL CERTIFICATE</td>
      <td>4.9</td>
      <td>150000</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Universidade de São Paulo</td>
      <td>SPECIALIZATION</td>
      <td>4.9</td>
      <td>4500</td>
    </tr>
  </tbody>
</table>
</div>

From a table we can see which organizations have the highest rating over certification type. Base by calculations all organizations have 4.9 rating base by student count.

### Calculate probability using Bayes's Theorem

As we know most popular difficulty is **Beginner**, given these criteria we can calculate **Baye's Theorem**. We will see that probability is to pick different course type over this difficulty. 

Bayes's Theorem P(A|B) = P(B|A)*P(A)/P(B)

Where:
<ol>
    <li>P(A∣B) is the probability of event A given that event B has occurred.</li>
    <li>P(B∣A) is the probability of event B given that event A has occurred.</li>
    <li>P(A) and P(B) are the probabilities of events A and B, respectively.</li>
</ol>


```python
# Calculate probabilities
total_courses = len(df)
p_course = len(df[df['course_Certificate_type'] == 'COURSE']) / total_courses
p_specialization = len(df[df['course_Certificate_type'] == 'SPECIALIZATION']) / total_courses
p_professional_certificate = len(
    df[df['course_Certificate_type'] == 'PROFESSIONAL CERTIFICATE']) / total_courses

p_course_given_beginner = len(
    df[(df['course_Certificate_type'] == 'COURSE') & (df['course_difficulty'] == 'Beginner')]) /
    len(df[df['course_difficulty'] == 'Beginner']
)
p_specialization_given_beginner = len(
    df[(df['course_Certificate_type'] == 'SPECIALIZATION') & (df['course_difficulty'] == 'Beginner')]) /
    len(df[df['course_difficulty'] == 'Beginner']
)
p_professional_certificate_given_beginner = len(
    df[(df['course_Certificate_type'] == 'PROFESSIONAL CERTIFICATE') & (df['course_difficulty'] == 'Beginner')]) /
    len(df[df['course_difficulty'] == 'Beginner']
)

print(f"P(COURSE | Beginner) = {p_course_given_beginner}")
print(f"P(SPECIALIZATION | Beginner) = {p_specialization_given_beginner}")
print(f"P(PROFESSIONAL CERTIFICATE | Beginner) = {p_professional_certificate_given_beginner}")
```

    P(COURSE | Beginner) = 0.5790554414784395
    P(SPECIALIZATION | Beginner) = 0.4024640657084189
    P(PROFESSIONAL CERTIFICATE | Beginner) = 0.018480492813141684
    

- P("COURSE" | "Beginner")
- P("SPECIALIZATION" | "Beginner")
- ("PROFESSIONAL CERTIFICATE" | "Beginner")
  
are the probabilities that a course belongs to each category given that it has a "Beginner" difficulty level.

### Analyse data using Simpson's paradox

Simpson's Paradox is a statistical phenomenon in which trends can appear in several groups of data but can disappear then groups are combined. For briew view let's analyze course ratings grouped by certificate types and difficulty.


```python
# Get average ratings for certification types
avg_ratings_by_certificate = df.groupby('course_Certificate_type')['course_rating'].mean()

# Calculate average ratings for all difficulty levels
avg_ratings_by_difficulty = df.groupby('course_difficulty')['course_rating'].mean()
```

Average Ratings by Certificate Type


```python
avg_ratings_by_certificate
```




    course_Certificate_type
    COURSE                      4.707045
    PROFESSIONAL CERTIFICATE    4.700000
    SPECIALIZATION              4.618182
    Name: course_rating, dtype: float64



Average Ratings by Difficulty Level


```python
avg_ratings_by_difficulty
```




    course_difficulty
    Advanced        4.600000
    Beginner        4.680903
    Intermediate    4.646465
    Mixed           4.708556
    Name: course_rating, dtype: float64



We can see that "COURSE" type have bigger rating over other types, also the "Mixed" difficulty is had higher ratings over others. But then we combine these data we can see different trend of Overall Average Rating:


```python
overall_avg_rating = df['course_rating'].mean()
overall_avg_rating
```




    4.6773288439955065



We see that overall rating is different from each calculated average ratings for certification types and difficulty levels. This is how Simpson's Paradox work, groups of data can differ on different group level.

### Summary
Overall **COURSE** certification type is the most popular, as we can understand **COURSE** can take lowest time to complete, students tend to finish it faster, by difficulty they pick **Beginner**, so we can assume students with less or no experience pick courses which are easy and fast to finish. Base by ratings students evaluate them quite well, most of the ratings goes from 4.1 to 5.0. Also, we saw that courses with over million students enrollment size takes big place in our data with 10.5% over all courses. We found most popular organizations, the top one is **University of Michigan** with more than 7 million students enrolled. As displayed before we saw that courses about **Machine Learning**, **Data Science** and **Python** are most popular and appear more often than others, so we assume that most popular courses is programming, 4 out of 5 top courses are base on programming. As evidence by using **Baye's Theorem** we check that probability to take **Course** which is **Beginner** level is about 58%.
### Suggestions for Improvement
- **Machine Learning Models:** If possible use MLM models to investigate data for predictive analysis, classification, etc. Models can help achieve new insights of data.
- **Bigger Dataset:** Having bigger data can help analyze it better, it will give more confidence over it.
- **More data features:** Will halp to analyse differently over genders, course price or course length.


### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact Information
[Email](ricardas.poskrebysev@gmail.com)
[LinkedIn](https://www.linkedin.com/in/ri%C4%8Dardas-poskreby%C5%A1evas-665207206/)
[GitHub](https://github.com/Riciokzz)

