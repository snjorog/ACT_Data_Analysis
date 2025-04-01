CBHS ACT Data

## Project: CBHS ACT Data Analysis (2021/22)
The goals of this analysis are to:
1.	Conduct descriptive statistics of all students that took the ACT in 22/23 school year.
2.	Conduct statistics across sub-groups.
3.	Find students meeting/not meeting ACT benchmarks.
4.  Conduct an analysis of math sub-scores.
5.	Determine which reporting categories have the greatest/least impact on Math composite score.
6.	Conduct analysis across ethnicities.
7.	Conduct analysis to determine how math sub score relate to the math classes they took.

Research Questions.
1.	What are our ACT Composite and ACT super score distributions (SOCS)?
2.	What are the sub scores and their distibutions?
3.	What percent of students are meeting the ACT sub-score benchmarks?
4.	What are the summary statistics of rthe 8 math reproting categories?
5.  How do the reporting categories impact the overall math score individually and collectively?
6.	What is the performances across ethnicities.
7.  What are the problems areas per subject? 

Plan of action
1.	Data scrapping from credible sources.
2.	Data cleaning and Data de-identification.
3.	Convert data to .csv and imported to a Pandas DataFrame. 
4.	Merge into a singular DataFrame. 
5.	Perform xploratory data analysis to answer RQs 1-7 above.
6.	Create visualization to present findings as needed.
7.  Draw  conclusion/inferences


```python
# imports
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import csv

# read csv file
#r= pd.read_csv("CBHS_ACT@21_22.csv")#21/22

#r= pd.read_csv("CBHS_21_22.csv")#21/22 with demographics

r= pd.read_csv("cbhs_22_23.csv")#22/23 with demographics
df= pd.DataFrame(r)

#print(df)
#df[['ACT composite score']].astype(int)
act_comp=df['ACT composite score']
#df['ACT composite score']=act_comp.astype(int) 
#act_comp=act_comp.astype(int)
print(act_comp.describe())


```

    count    299.000000
    mean      22.086957
    std        5.365731
    min       12.000000
    25%       18.000000
    50%       22.000000
    75%       26.000000
    max       35.000000
    Name: ACT composite score, dtype: float64
    

1.	What are our ACT Composite and ACT super score distributions (SOCS)?


```python
print(act_comp.describe())
df.dtypes
```

    count    299.000000
    mean      22.086957
    std        5.365731
    min       12.000000
    25%       18.000000
    50%       22.000000
    75%       26.000000
    max       35.000000
    Name: ACT composite score, dtype: float64
    




    ACT ID                                         int64
    Last Updated (CT)                             object
    Test Date                                     object
    Ethnicity                                     object
    Average GPA                                  float64
                                                  ...   
    State rank of english language arts score     object
    National Career Readiness Certificate         object
    State Org Number                               int64
    District Org Number                            int64
    School Org Number                              int64
    Length: 129, dtype: object




```python
#Percent meeting Compsite Benchmark
met=0
not_met=0
for s in act_comp:
    if s>=20.6:
        met+=1
    else:
        not_met+=1
print(met)
print(not_met)
per=met/(met+not_met)
print(per)
```

    173
    127
    0.5766666666666667
    


```python
#Plot the ACT Composite Histogram
nat_mean=20.8
plt.hist(act_comp, color='lightblue')
plt.axvline(act_comp.mean(),color='green', label='CBHS Mean Composite Score' ) #Shows the mean line
plt.axvline(nat_mean,color='r', label='National Mean Composite Score' )
plt.title("CBHS 22/23 ACT Data")
plt.legend(bbox_to_anchor=(1.0,1), loc='upper left')
plt.show()
```


    
![png](output_6_0.png)
    


Describe the Histogram (SOCS)
1.Close to norma, with a slight left skew due students who scored 33-36
2.No outliers
3.Center at 23.61
4.Average variation from the mean is 5.12


```python
#Plot the ACT Composite Box Plot
plt.boxplot(act_comp)
plt.title("CBHS 22/23 ACT Data")
plt.show()

#Plot the ACT Math Box Plot
plt.boxplot(act_math)
plt.title("CBHS 21/22 ACT Math Data")
plt.show()
```


    
![png](output_8_0.png)
    



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-eb6a76d0549c> in <module>
          5 
          6 #Plot the ACT Math Box Plot
    ----> 7 plt.boxplot(act_math)
          8 plt.title("CBHS 21/22 ACT Math Data")
          9 plt.show()
    

    NameError: name 'act_math' is not defined


No outlires

What are the sub scores and their distibutions?

DISTRBUTION OF SCORES (below 20, 20-24, 25-29, 30-36)


```python
def count_items_in_ranges(alist):
    cat1=0
    cat2=0
    cat3=0
    cat4=0
    for item in alist:
        if item<20:
            cat1+=1
        if item>=20 and item<=24:
            cat2+=1
        if item>=22 and item<=29:
            cat3+=1
        if item>=30:
            cat4+=1
    return cat1,cat2,cat3,cat4

cat1,cat2,cat3,cat4=count_items_in_ranges(act_comp)
data={"Below 20":cat1, "20-24":cat2, "25-29":cat3,"above 30":cat4}
cats=list(data.keys())
vals=list(data.values())
c = ['red', 'yellow', 'blue', 'orange']           
plt.bar(cats, vals, color=c)
plt.title("ACT Composite Score Distribution")
for y, x in zip(vals, cats):
    plt.annotate(f'{y}\n', xy=(x, y), ha='center', va='center')
plt.show()
```


    
![png](output_12_0.png)
    



```python
fig, ax =plt.subplots(1,1)
data=[[cat1,cat2,cat3,cat4]]
column_labels=["Below 20", "20-24", "25-29", "30-36"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")

plt.show()
```


    
![png](output_13_0.png)
    


ACT MATH


```python
act_math=df['ACT math score']
#print(act_math)
```


```python
print(act_math.describe())
```

    count    300.000000
    mean      21.400000
    std        5.235587
    min       12.000000
    25%       16.000000
    50%       21.000000
    75%       25.250000
    max       35.000000
    Name: ACT math score, dtype: float64
    


```python
act_eng=df['ACT English score']
print(act_eng.describe())
```

    count    299.000000
    mean      21.602007
    std        6.245438
    min       10.000000
    25%       17.000000
    50%       21.000000
    75%       25.000000
    max       35.000000
    Name: ACT English score, dtype: float64
    


```python
#Plot the ACT Math Score Histogram
#Benchmark score
ben=22
plt.hist(act_math, color='lightblue')
plt.axvline(act_math.mean(),color='green', label='CBHS Mean Math Score' ) #Shows the mean line
plt.axvline(x=ben,color='r', label='ACT Math Benchmark Score') #Shows the mean line
plt.title("CBHS 22/23 ACT Math")
plt.legend(bbox_to_anchor=(1.0,1), loc='upper left')
plt.show()
```


    
![png](output_18_0.png)
    


ACT MATH College Readiness


```python

def count_readiness(nlist):
    met=0
    not_met=0

    for item in nlist:
        if item=="Met":
            met+=1
        if item=="Not Met":
            not_met+=1
        
    return met, not_met
met,not_met=count_readiness(df['ACT College Readiness reading benchmark'])
print("Met ",met)
print("Not met", not_met)
print(not_met/(met+not_met))

y = np.array([met, not_met])
mylabels=['Met','Not Met']
exp=[0.05,0.05]
plt.pie(y,labels=mylabels,autopct='%.f%%', explode=exp)
plt.title("Met/Not Met Reading benchmark")
#plt.legend()
plt.show()

```

    Met  161
    Not met 139
    0.4633333333333333
    


    
![png](output_20_1.png)
    


ACT Scores By Ethinicity


```python
                             
#Ethinicities:Black/African American, White, Asian,Hispanic/Latino,Two or more races, 
#Prefer not to respond, American Indian/Alaska Native,Native Hawaiian/Other Pacific Islander,Not provided

#Plot EthicityComposite Scores
mean_by_eth=df.groupby('Ethnicity')['ACT composite score'].mean()

mean_by_eth.plot(kind='barh', title='ACT Composite By Ethnicity', ylabel='Mean', figsize=(10, 5))
'''
for y, x in zip(s, c):
    plt.annotate(f'{y}\n', xy=(x, y), ha='center', va='center')
'''
print(mean_by_eth)
    



```

    Ethnicity
    American Indian/Alaska Native    16.000000
    Asian                            21.000000
    Black/African American           17.869565
    Hispanic/Latino                  19.904762
    Not Provided                     16.500000
    Prefer not to respond            20.000000
    Two or more races                21.500000
    White                            23.219626
    Name: ACT composite score, dtype: float64
    


    
![png](output_22_1.png)
    



```python
#eth_comp(df)
```


```python
#Plot the ACT Math Box Plot
plt.boxplot(act_math)
plt.title("CBHS 21/22 ACT Math Data")
plt.show()
```


    
![png](output_24_0.png)
    


Right skewed

Describe the Histogram (SOCS)
1.Right skew due students who scored 33-36
2.No outliers
3.Center at 22.47. Mean is slighyly above national benchmark score of 22.
4.Variation from the mean is 5.47

3. Percentage of Students below Benchmark


```python

def percent_below_benchmark(score_list,ben):
    '''
    Calculates percent above benchmark
    '''
    above_count=sum(1 for score in score_list if score>ben)
    return 100-above_count/len(score_list)*100
```


```python
percent_below_benchmark(act_math, ben)
```




    57.33333333333333



51.22% of students not meeting ACT benchmark in math.


```python

```

ACT ENG


```python
act_eng=df['ACT English score']
print(act_eng.describe())
```

    count    299.000000
    mean      21.602007
    std        6.245438
    min       10.000000
    25%       17.000000
    50%       21.000000
    75%       25.000000
    max       35.000000
    Name: ACT English score, dtype: float64
    


```python
#Plot the ACT English Score Histogram
#Benchmark score
ben_eng=18
plt.hist(act_eng, color='lightblue')
plt.axvline(act_eng.mean(),color='green', label='Mean score' ) #Shows the mean line
plt.axvline(x=ben_eng,color='r', label='ACT English Benchmark') #Shows the mean line
plt.title("CBHS 21/22 ACT English")
plt.legend(bbox_to_anchor=(1.0,1), loc='upper left')
plt.show()
```


    
![png](output_34_0.png)
    



```python
#Plot the ACT English Box Plot
plt.boxplot(act_eng)
plt.title("CBHS 21/22 ACT English Scores")
plt.show()
```


    
![png](output_35_0.png)
    


One outlier(7.0)

Describe the Histogram (SOCS)
1.Left skew due students who scored below 12
2.One outliers
3.Center at 23.76. Mean is above national benchmark score of 18.
4.Variation from the mean is 6.22


```python
percent_below_benchmark(act_eng, ben_eng)
```




    33.33333333333334



20% of students not meeting ACT benchmark in math.

ACT Science


```python
act_sci=df['ACT science score']
print(act_sci.describe())
```

    count    300.000000
    mean      22.216667
    std        5.411779
    min        8.000000
    25%       19.000000
    50%       22.000000
    75%       25.000000
    max       36.000000
    Name: ACT science score, dtype: float64
    


```python
#Plot the ACT science Score Histogram
#Benchmark score
ben_sci=23
plt.hist(act_sci, color='lightblue')
plt.axvline(act_sci.mean(),color='green', label='Mean Science Score' ) #Shows the mean line
plt.axvline(x=ben_sci,color='r', label='ACT Science Benchmark Score') #Shows the mean line
plt.title("22/23 Science Scores Distribution")
plt.legend(bbox_to_anchor=(1.0,1), loc='upper left')
plt.show()
```


    
![png](output_42_0.png)
    



```python
#Plot the ACT Science Box Plot
plt.boxplot(act_sci)
plt.title("CBHS 21/22 ACT Science Scores")
plt.show()
```


    
![png](output_43_0.png)
    


Describe the Histogram (SOCS) 
1.No skew.
2.Two outliers 
3.Center at 23.22. Mean is slightly above national benchmark score of 23. 
4.Variation from the mean is 5.15


```python
percent_below_benchmark(act_sci, ben_sci)
```




    64.0



55.79% of students not meeting ACT benchmark in science.

ACT READING


```python
act_rd=df['ACT reading score']
print(act_rd.describe())
```

    count    300.000000
    mean      22.576667
    std        6.662739
    min        9.000000
    25%       18.000000
    50%       22.000000
    75%       26.000000
    max       36.000000
    Name: ACT reading score, dtype: float64
    


```python
#Plot the ACT Reading Score Histogram
#Benchmark score
ben_rd=22
plt.hist(act_rd, color='lightblue')
plt.axvline(act_rd.mean(),color='green', label='CBHS Mean Reading Score' ) #Shows the mean line
plt.axvline(x=ben_rd,color='r', label='ACT Science Reading Score') #Shows the mean line
plt.title("22/23 ACT Reading Scores")
plt.legend(bbox_to_anchor=(1.0,1), loc='upper left')
plt.show()
```


    
![png](output_49_0.png)
    



```python
#Plot the ACT Reading Box Plot
plt.boxplot(act_rd)
plt.title("CBHS 21/22 ACT Reading Scores")
plt.show()
```


    
![png](output_50_0.png)
    


Describe the Histogram (SOCS) 
1.No skew.
2.No outliers 
3.Center at 24.53. Mean is above national benchmark score of 22. 
4.Variation from the mean is 6.11


```python
percent_below_benchmark(act_rd, ben_rd)
```




    51.666666666666664



41.32% of students not meeting ACT benchmark in reading.

SUMMARY


```python
#Bar plot of ACT Mean Sub-scores
c = np.array(["ACT Composite","Math", "English", "Reading", "Science"])
s = np.array([23.61, 22.4, 23.76, 24.53, 23.21])
for y, x in zip(s, c):
    plt.annotate(f'{y}\n', xy=(x, y), ha='center', va='center')
plt.bar(c,s, color="#00bfff")
plt.title("22/23 ACT Mean Composite & Sub-scores")

plt.show()
```


    
![png](output_55_0.png)
    



```python
print(s)
print(c)
```

    [23.61 22.4  23.76 24.53 23.21]
    ['ACT Composite' 'Math' 'English' 'Reading' 'Science']
    


```python
#Bar plot of ACT Meeting/Not Meeting Benchmarks

import matplotlib.pyplot as plt

# Sample data
categories = ["Math", "English", "Reading", "Science"]
data1 = [50.7,28.7,46.7,53.3]
data2 = [49.3,71.3,53.3,44.7]

# Creating the stacked bar plot
fig, ax = plt.subplots()
ax.bar(categories, data1, label='Did not Meet Benchmark')
ax.bar(categories, data2, bottom=data1, label='Met Benchmark')
#ax.bar(categories, data3, bottom=[i+j for i,j in zip(data1, data2)], label='Data 3')

# Adding labels and legend
ax.set_ylabel('Percent')
ax.set_title('CBHS Benchmark Data')
ax.legend()

# Display the plot
plt.show()

```


    
![png](output_57_0.png)
    


Pie


```python
subscores = np.array(["Math", "English", "Reading", "Science"])
y = np.array(["51.22", "33.33","51.67", "55.79"])
exp=[0.05,0.05,0.05,0.05]
plt.pie(y, labels=subscores, autopct='%.f%%', explode=exp)
plt.title("Students Not Meeting Benchmarks")
#plt.legend()
plt.show()

```


    
![png](output_59_0.png)
    


MATH Reporting Categories.


```python
#Bar plot MATH Reporting Categories
x = np.array(['Number & Quantity Percent Correct','Algebra Percent Correct','Functions Percent Correct'
,'Geometry Percent Correct','Statistics & Probability Percent Correct'
,'Integrating Essential Skills Percent Correct','Modeling Percent Correct'])
nq=df['Number & Quantity Percent Correct'].mean()
a=df['Algebra Percent Correct'].mean()
f=df['Functions Percent Correct'].mean()
g=df['Geometry Percent Correct'].mean()
sp=df['Statistics & Probability Percent Correct'].mean()
i=df['Integrating Essential Skills Percent Correct'].mean()
m=df['Modeling Percent Correct'].mean()

y=[nq,a,f,g,sp,i,m]

plt.barh(x,y, color="#00bfff")
plt.title("ACT MATH Reporting Categories")
plt.show()
```


    
![png](output_61_0.png)
    



```python
#Bar plot MATH Reporting Categories Impact on MATH Composite
x = np.array(['Number & Quantity Percent Correct','Algebra Percent Correct','Functions Percent Correct'
,'Geometry Percent Correct','Statistics & Probability Percent Correct'
,'Integrating Essential Skills Percent Correct','Modeling Percent Correct'])
nqi=0.01891001 
ai=0.03393799
fi=0.03519358 
gi=0.04522195
spi=0.01960279
ii=0.10649198
mi=0.01531777


y=[nqi,ai,fi,gi,spi,ii,mi]

plt.barh(x,y, color="#00bfff")
plt.title("ACT MATH Categories Impact on MATH Composite")
plt.show()
```


    
![png](output_62_0.png)
    


Numbers & Quantity



```python
nq=df['Number & Quantity Percent Correct']
print(nq.describe())
```

    count    300.000000
    mean      58.613333
    std       26.849708
    min        0.000000
    25%       33.000000
    50%       67.000000
    75%       83.000000
    max      100.000000
    Name: Number & Quantity Percent Correct, dtype: float64
    


```python
#Plot the ACT Reading Score Histogram
plt.hist(nq, color='lightblue')
plt.axvline(nq.mean(),color='green', label='Number and Quatity Mean' ) #Shows the mean line
plt.title("Number and Quantiry Percent Correct")
plt.legend(bbox_to_anchor=(1.0,1), loc='upper left')
plt.show()
```


    
![png](output_65_0.png)
    



```python
#Numbers and Quantity Box Plot
plt.boxplot(nq)
plt.title("Number and Quantity Percent Correct")
plt.show()
```


    
![png](output_66_0.png)
    


MULTIPLE REGRESSION FOR MATH CATEGORIES


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read data from CSV file
r= pd.read_csv("CBHS_ACT@21_22.csv")
data= pd.DataFrame(r)

#x1=data['Number & Quantity Percent Correct']
#x2=data["Algebra Percent Correct"]

# Split the data into features (X) and target variable (y)
x = data[['Number & Quantity Percent Correct','Algebra Percent Correct','Functions Percent Correct'
,'Geometry Percent Correct','Statistics & Probability Percent Correct'
,'Integrating Essential Skills Percent Correct','Modeling Percent Correct']]
y = data['ACT math score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)

```

    Coefficients: [0.01891001 0.03393799 0.03519358 0.04522195 0.01960279 0.10649198
     0.01531777]
    Intercept: 7.241542731098839
    Mean Squared Error: 0.7425299151028774
    

SUMMARY OF MUTIPLE REGRESSION

COEFFICIENTS
Coeffcients gives the size of the effect of the independent varable on the dependent variable.

Number & Quantity Percent Correct (0.01891001)
Algebra Percent Correct(0.03393799)
Functions Percent Correct(0.03519358)
Geometry Percent Correct(0.04522195)
Statistics & Probability Percent Correct (0.01960279)
Integrating Essential Skills Percent Correct(0.10649198)
Modeling Percent Correct(0.01531777)


1.Geometry Percent Correct(0.04522195)
2.Functions Percent Correct(0.03519358)
3.Algebra Percent Correct(0.03393799)





```python
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
reg.coef_
```




    array([0.5, 0.5])



TRENDS


```python
import matplotlib.pyplot as plt
import numpy as np

r= pd.read_csv("trends.csv")#2019-2023
tr= pd.DataFrame(r)
#print(tr)

#plot 1:
year=np.array(["19-20","20-21", "21-22", "22-23", "23-24"])
comp=tr['Composite Valid Number']
comp = list(reversed(comp))

# making subplots

#plt.subplot(3, 3, 1)
plt.bar(year,comp)
plt.xlabel("Year")
plt.ylabel("Mean Score")
plt.title("Valid Number of Testers 5-Year Trend")
for y, x in zip(comp, year):
    plt.annotate(f'{y}\n', xy=(x, y), ha='center', va='center')

plt.show()


```


    
![png](output_72_0.png)
    



```python
print(comp)
print(year)
```

    [24.9, 24.3, 24.5, 22.9, 22.6]
    ['19-20' '20-21' '21-22' '22-23' '23-24']
    

1. Assessment of Student Performance:
  - Composite scores are clustered around the range of 20 to 25, suggesting a relatively normal distribution. 
  -218 students below a score of 20 in year 2022/23.

2. Identification of Achievement Gaps:
         - Math sub scores are lowest.
         - Black/African underperforming as compared to other ethnicities.

3. College Readiness Measurement:
         - 42% of students not meeting sub-score college readiness benchmarks in math and science

4. Analysis of Subjects Reporting Categories: Students struggle most in
         -Math: Functions
         -Science: Scientific Investigation
         -English: Knowledge of Language 
         -Reading: Integration of Knowledge and Ideas
5. Longitudinal Analysis: 5-Year Trend (2019/20 - 2022/23)
         - Trends show a decrease in mean score over the last five-years.
                - Mean Composite: 6.04% decrease.
                - Mean Math:         6.96% decrease.
                - Mean English:      8.68 % decrease. 
                - Mean Reading:    8.03% decrease. 
                - Mean Science:      3.69 % decrease. 


Limitations
1.Changing test formats: changes to the ACT test format or structure during the five-year might impact the comparability of  data across years.
2.Participation rates: Fluctuations in the participation rates across different years can introduce biases in the data.
3.External factors: External factors, for example COVID lockdowns.
4.Comparison data from the NCES reports the highest score a student got in a given year
5.Access ACT(CBHS data) reports all scores taken each year and the super score.

