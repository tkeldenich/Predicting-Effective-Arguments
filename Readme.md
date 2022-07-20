# **EDA on NLP How to Do It ? – Best Tutorial Simple**

- [English Article](https://inside-machinelearning.com/en/eda-nlp/)
- [French Article](https://inside-machinelearning.com/eda-sur-nlp/)

Today, let’s see together how to apply Exploratory Data Analysis (EDA) on a NLP dataset: feedback-prize-effectiveness.

In a previous article we saw how to do EDA to [prevent environmental disasters.](https://inside-machinelearning.com/en/data-science-and-environment/) Thanks to meteorological data, we analyzed [the causes of forest fires.](https://inside-machinelearning.com/en/data-science-and-environment/)

**Here I propose to apply the same analysis on text data thanks to the** [feedback-prize-effectiveness](https://github.com/tkeldenich/-Predicting-Effective-Arguments) **dataset.**

This dataset is taken from the Kaggle competition of the same name: [Feedback Prize – Predicting Effective Arguments.](https://www.kaggle.com/competitions/feedback-prize-effectiveness/overview)

The main objective of this competition is to create a Machine Learning algorithm able to predict the effectiveness of a discourse. Here we will see in detail the Exploratory Data Analysis (EDA) that will allow us to understand our dataset.

## **Data**

First thing to do, download the dataset. Either by registering to the [Kaggle](https://www.kaggle.com/competitions/feedback-prize-effectiveness/overview) contest, or by downloading it on [this Github.](https://github.com/tkeldenich/-Predicting-Effective-Arguments)

Then you can open it with Pandas and display dimensions:


```python
import pandas as pd

df = pd.read_csv("train.csv")
df.shape
```




    (36765, 5)



We have 36.765 rows so 36.765 discourses for 5 columns.

Now let’s see what these columns represent by displaying their types:


```python
df.dtypes
```




    discourse_id               object
    essay_id                   object
    discourse_text             object
    discourse_type             object
    discourse_effectiveness    object
    dtype: object



I display here the columns and their description:

- `discourse_id` – object – discourse ID
- `essay_id` – object – ID of the essay (an essay can be composed of several discourses)
- `discourse_text` – object – Discourse text
- `discourse_type` – object – Type of discourse
- `discourse_effectiveness` – object – Effectiveness of the discourse

2 ID columns, one column representing text and 2 columns for Labels. The one we are most interested in is `discourse_effectiveness` as it is the target to predict.

Then we can display our data:


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discourse_id</th>
      <th>essay_id</th>
      <th>discourse_text</th>
      <th>discourse_type</th>
      <th>discourse_effectiveness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0013cc385424</td>
      <td>007ACE74B050</td>
      <td>Hi, i'm Isaac, i'm going to be writing about h...</td>
      <td>Lead</td>
      <td>Adequate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9704a709b505</td>
      <td>007ACE74B050</td>
      <td>On my perspective, I think that the face is a ...</td>
      <td>Position</td>
      <td>Adequate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c22adee811b6</td>
      <td>007ACE74B050</td>
      <td>I think that the face is a natural landform be...</td>
      <td>Claim</td>
      <td>Adequate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a10d361e54e4</td>
      <td>007ACE74B050</td>
      <td>If life was on Mars, we would know by now. The...</td>
      <td>Evidence</td>
      <td>Adequate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>db3e453ec4e2</td>
      <td>007ACE74B050</td>
      <td>People thought that the face was formed by ali...</td>
      <td>Counterclaim</td>
      <td>Adequate</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have a good overview of the dataset, we can start EDA!

**To do this, we’ll go through the classic steps of the process:**

- Understand our dataset with the **Univariate Analysis**
- Drawing hypotheses with the **Multivariate Analysis**

## **Univariate Analysis**

Univariate analysis is the fact of examining each feature separately.

This will allow us to get a deeper understanding of the dataset.

> Here, we are in the comprehension phase.

**The question associated with the Univariate Analysis is: What is the characteristics of the data that compose our dataset?**



### **Target**

As we noticed above, the most interesting column for us is the target `discourse_effectiveness`. This column indicates the effectiveness of a discourse.

Each line, each discourse, can have a different effectiveness. We classify them according to 3 levels:

- `Ineffective`
- `Adequate`
- `Effective`

Let’s see how these 3 classes are distributed in our dataset:


```python
import seaborn as sns
import matplotlib.pyplot as plt

stats_target = df['discourse_effectiveness'].value_counts(normalize=True)
print(stats_target)

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.countplot(data=df,y='discourse_effectiveness')
plt.subplot(1,2,2)
stats_target.plot.bar(rot=25)
plt.ylabel('discourse_effectiveness')
plt.xlabel('% distribution per category')
plt.tight_layout()
plt.show()
```

    Adequate       0.570570
    Effective      0.253665
    Ineffective    0.175765
    Name: discourse_effectiveness, dtype: float64



![png](Readme_files/Readme_15_1.png)


**Here you can see the numerical distribution, but also the statistical one which is easier to analyze.**

57% of the discourses are `Adequate`, the rest are either `Effective` or `Ineffective`.

Ideally, we would have needed more `Ineffective` discourses to have a better balanced and therefore generalizable dataset. Since we have no influence on the data, let’s continue with what we have!

### **Categorical Data**

Now I propose to analyze the types of discourses.

There are seven types:

- **Lead** – an introduction that begins with a statistic, quote, description, or other means of getting the reader’s attention and directing them to the thesis
- **Position** – an opinion or conclusion on the main issue
- **Claim** – a statement that supports the position
**Counterclaim** – a claim that refutes another claim or gives a reason opposite to the position
- **Rebuttal** – a statement that refutes a counterclaim
- **Evidence** – ideas or examples that support assertions, counterclaims or rebuttals
- **Concluding Statement** – a final statement that reaffirms the claims

Given the different types, it would seem logical that there are fewer `Counterclaims` and `Rebuttal` than other types.

**Furthermore, I would like to remind here that multiple discourses, multiple lines, can be part of the same essay (`essay_id`). That is, multiple discourses can be written by the same author, in the same context. And so an `essay` can contain multiple `discourse` having different types as well as different degrees of effectiveness.**

Let us now analyze the distribution of this `discourse_type`:




```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.countplot(data=df,y='discourse_type')
plt.subplot(1,2,2)
df['discourse_type'].value_counts(normalize=True).plot.bar(rot=25)
plt.ylabel('discourse_type')
plt.xlabel('% distribution per category')
plt.tight_layout()
plt.show()
```


![png](Readme_files/Readme_19_0.png)


We can see here that the distribution is not equally distributed. Our hypothesis is verified for `Counterclaim` and `Rebuttal`. Nevertheless, the distribution is extremely unbalanced in favor of `Claim` and `Evidence`. Let’s keep that in mind for further analysis.

### **NLP Data**

#### **Discourse Length**

Now that we have analyzed the categorical data. We can move on to analyze the NLP data.

First, let’s analyze the length of our sentences.

To do this, we create a new column `discourse_length` containing the size of each discourse:


```python
def length_disc(discourse_text):
    return len(discourse_text.split())

df['discourse_length'] = df['discourse_text'].apply(length_disc)
```

Let’s display the result:


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discourse_id</th>
      <th>essay_id</th>
      <th>discourse_text</th>
      <th>discourse_type</th>
      <th>discourse_effectiveness</th>
      <th>discourse_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0013cc385424</td>
      <td>007ACE74B050</td>
      <td>Hi, i'm Isaac, i'm going to be writing about h...</td>
      <td>Lead</td>
      <td>Adequate</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9704a709b505</td>
      <td>007ACE74B050</td>
      <td>On my perspective, I think that the face is a ...</td>
      <td>Position</td>
      <td>Adequate</td>
      <td>41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c22adee811b6</td>
      <td>007ACE74B050</td>
      <td>I think that the face is a natural landform be...</td>
      <td>Claim</td>
      <td>Adequate</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a10d361e54e4</td>
      <td>007ACE74B050</td>
      <td>If life was on Mars, we would know by now. The...</td>
      <td>Evidence</td>
      <td>Adequate</td>
      <td>72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>db3e453ec4e2</td>
      <td>007ACE74B050</td>
      <td>People thought that the face was formed by ali...</td>
      <td>Counterclaim</td>
      <td>Adequate</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



Now we can examine the `discourse_length` column like any other numeric data:


```python
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.kdeplot(df['discourse_length'],color='g',shade=True)
plt.subplot(1,2,2)
sns.boxplot(df['discourse_length'])
plt.show()
```

    /Users/tomkeldenich/opt/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(



![png](Readme_files/Readme_28_1.png)


t looks like there are a lot of values that are extremely far from the mean. These are called outliers and they impact our analysis. We can’t properly breakdown the Tukey box on the right.

**Let’s zoom in on the graph:**


```python
plt.figure(figsize=(16,5))
ax = sns.boxplot(df['discourse_length'])
ax.set_xlim(-1, 200)
plt.show()
```

    /Users/tomkeldenich/opt/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(



![png](Readme_files/Readme_30_1.png)


That’s better! Most discourses are less than 120 words long, with the average being about 30 words.

**Despite this, it seems that there are many discourses above 120 words. Let’s analyze these outliers.**

First, by calculating [Skewness and Kurtosis.](https://inside-machinelearning.com/en/skewness-and-kurtosis/) Two measures, [detailed in this article](https://inside-machinelearning.com/en/skewness-and-kurtosis/), that help us to understand outliers and their distribution:


```python
print("Skew: {}".format(df['discourse_length'].skew()))
print("Kurtosis: {}".format(df['discourse_length'].kurtosis()))
```

    Skew: 2.9285198590138415
    Kurtosis: 15.61832354148293


The outliers are widely spaced from the average with a very high Kurtosis.

##### **Outliers**

In most distributions, it is normal to have extreme values.

**But outliers are not very frequent, even anomalous. It can be an error in the dataset.**

Therefore, let’s display these outliers to determine if it is an error.

To determine outliers, we use the z-score.

> Z-score calculates the distance of a point from the mean.

**If the z-score is less than -3 or greater than 3, it is considered an outlier.**

Let’s see this by displaying all points below -3 and above 3:


```python
from scipy.stats import zscore

y_outliers = df[abs(zscore(df['discourse_length'])) >= 3 ]
y_outliers
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discourse_id</th>
      <th>essay_id</th>
      <th>discourse_text</th>
      <th>discourse_type</th>
      <th>discourse_effectiveness</th>
      <th>discourse_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>4a7d11406893</td>
      <td>01E9D9CD5CBF</td>
      <td>The study the ability of humans to read subatl...</td>
      <td>Evidence</td>
      <td>Ineffective</td>
      <td>222</td>
    </tr>
    <tr>
      <th>110</th>
      <td>d1c5f0d13151</td>
      <td>021663FD2F2E</td>
      <td>The Mona lisa demonstration really intended to...</td>
      <td>Evidence</td>
      <td>Ineffective</td>
      <td>209</td>
    </tr>
    <tr>
      <th>208</th>
      <td>4b1e4c493bfd</td>
      <td>0491C7BFA9B4</td>
      <td>Attention !!! to all the residents of this com...</td>
      <td>Evidence</td>
      <td>Ineffective</td>
      <td>353</td>
    </tr>
    <tr>
      <th>219</th>
      <td>1b263824b0b2</td>
      <td>04B4209D8A34</td>
      <td>Even if you have no experience with an of the ...</td>
      <td>Evidence</td>
      <td>Effective</td>
      <td>199</td>
    </tr>
    <tr>
      <th>293</th>
      <td>506f1d68d554</td>
      <td>071BF63AF332</td>
      <td>ubs and sports are very social activities. If ...</td>
      <td>Evidence</td>
      <td>Effective</td>
      <td>249</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36577</th>
      <td>5c7052b1ac08</td>
      <td>F359E262A14A</td>
      <td>That is so important for me and my parents and...</td>
      <td>Evidence</td>
      <td>Ineffective</td>
      <td>234</td>
    </tr>
    <tr>
      <th>36620</th>
      <td>e6613c2ffde8</td>
      <td>F54BD89B665A</td>
      <td>Online courses do not give the opportunity for...</td>
      <td>Evidence</td>
      <td>Effective</td>
      <td>192</td>
    </tr>
    <tr>
      <th>36654</th>
      <td>b1f96b34280e</td>
      <td>F93D06BC99D8</td>
      <td>Generic_Name found out much later that she mad...</td>
      <td>Concluding Statement</td>
      <td>Effective</td>
      <td>257</td>
    </tr>
    <tr>
      <th>36689</th>
      <td>479fb02ae14b</td>
      <td>FD05FDCEA11B</td>
      <td>My footsteps seemed impossibly loud as I walke...</td>
      <td>Lead</td>
      <td>Effective</td>
      <td>542</td>
    </tr>
    <tr>
      <th>36722</th>
      <td>dc9995b62bb6</td>
      <td>FDF0AEEB14C3</td>
      <td>In conclusion, many people will argue that onl...</td>
      <td>Concluding Statement</td>
      <td>Effective</td>
      <td>215</td>
    </tr>
  </tbody>
</table>
<p>719 rows × 6 columns</p>
</div>



719 lines are outliers. They do not seem to represent errors. We can consider these lines as discourses that have not been separated in multiple essays (to be checked).

Let’s display the distribution of the efficiency of these outliers:


```python
stats_long_text = y_outliers['discourse_effectiveness'].value_counts(normalize=True)
print(stats_long_text)
stats_long_text.plot.bar(rot=25)
```

    Effective      0.536857
    Ineffective    0.342142
    Adequate       0.121001
    Name: discourse_effectiveness, dtype: float64





    <AxesSubplot:>




![png](Readme_files/Readme_38_2.png)


Here, a first hint emerges. Most long discourses seem to be `Effective` at about 53%. This is much more than in the whole dataset (25%).

We can therefore formulate a first hypothesis: **the longer a discourse is, the more `Effective` it seems.**

But we can also see that it can be more frequently `Ineffective` (34%) than a discourse of normal length (17%).

Let us display the distribution of types for these outliers:


```python
stats_long_text = y_outliers['discourse_type'].value_counts(normalize=True)
print(stats_long_text)
stats_long_text.plot.bar(rot=25)
```

    Evidence                0.933241
    Concluding Statement    0.034771
    Lead                    0.029207
    Counterclaim            0.001391
    Rebuttal                0.001391
    Name: discourse_type, dtype: float64





    <AxesSubplot:>




![png](Readme_files/Readme_40_2.png)


On this point, the statistic is very clear. Most long discourses are `Evidence`.

But are most `Evidence` discourses `Effective`?

We will see this in the Multivariate Analysis.

For now, let’s continue with the analysis of the words that make up the discourses!

#### **Preprocessing**

In order to analyze the words that compose the discourses, we will first perform a preprocessing by removing:

- numbers
- stopwords
- special characters

Here I copy and paste the code from the [Preprocessing NLP – Tutorial to quickly clean up a text](https://inside-machinelearning.com/en/preprocessing-nlp-preprocessing/) and applying a few changes:


```python
import nltk
import string
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()

def preprocessSentence(sentence):
    sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])

    sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())

    tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)

    words_w_stopwords = [i for i in tokenize_sentence if i not in stopwords]

    words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)
    
    words_lemmatize = (re.sub(r"[^a-zA-Z0-9]","",w) for w in words_lemmatize)

    sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in words or not w.isalpha())

    return sentence_clean.split()
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/tomkeldenich/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/tomkeldenich/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package words to
    [nltk_data]     /Users/tomkeldenich/nltk_data...
    [nltk_data]   Package words is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/tomkeldenich/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


As an example let’s display a base sentence and a preprocessed one:


```python
print(df.iloc[1]['discourse_text'])
print('\n')
print(preprocessSentence(df.iloc[1]['discourse_text']))
```

    On my perspective, I think that the face is a natural landform because I dont think that there is any life on Mars. In these next few paragraphs, I'll be talking about how I think that is is a natural landform 
    
    
    ['perspective', 'think', 'face', 'natural', 'dont', 'think', 'life', 'mar', 'next', 'paragraph', 'ill', 'talking', 'think', 'natural']


Now let’s apply the preprocessing to the whole DataFrame:


```python
df_words = df['discourse_text'].apply(preprocessSentence)
```

We get the result in `df_words`.

#### **Word Analysis**

We now have a DataFrame containing our preprocessed discourses. Each line represents a list containing the words composing discourses.

I would like to perform a one-hot encoding here. Again the process is explained in our article on [NLP Preprocessing.](https://inside-machinelearning.com/en/preprocessing-nlp-preprocessing/)

**The idea of one-hot encoding is to have columns representing every words in the dataset, and rows indicating 1 if the word is present in the discourse, 0 otherwise.**

If you have enough memory space use this line to make the one-hot encoding:


```python
#dfa = pd.get_dummies(df_words.apply(pd.Series).stack()).sum(level=0)
```

**Otherwise use the `concat` option by splitting your dataset in two(or more) and one hot encoding the parts separately.**

Finish by concatenating them:


```python
dfa1 = pd.get_dummies(df_words.iloc[:20000].apply(pd.Series).stack()).sum(level=0)
```

    /var/folders/bm/5ssydkrd4c1cpc7_7mjf7ym40000gn/T/ipykernel_1783/227132595.py:1: FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
      dfa1 = pd.get_dummies(df_words.iloc[:20000].apply(pd.Series).stack()).sum(level=0)



```python
dfa2 = pd.get_dummies(df_words.iloc[20000:].apply(pd.Series).stack()).sum(level=0)
```

    /var/folders/bm/5ssydkrd4c1cpc7_7mjf7ym40000gn/T/ipykernel_1783/1417337473.py:1: FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
      dfa2 = pd.get_dummies(df_words.iloc[20000:].apply(pd.Series).stack()).sum(level=0)



```python
dfb = pd.concat([dfa1,dfa2], axis=0, ignore_index=True)
```


```python
dfb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>aa</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abhor</th>
      <th>ability</th>
      <th>able</th>
      <th>aboard</th>
      <th>abolish</th>
      <th>abolishment</th>
      <th>...</th>
      <th>wrestle</th>
      <th>wristband</th>
      <th>writhen</th>
      <th>wrongful</th>
      <th>yah</th>
      <th>yell</th>
      <th>yote</th>
      <th>youthfulness</th>
      <th>zipping</th>
      <th>zoom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9561 columns</p>
</div>



If you have used the `concat` option, you need now to replace the NaN with 0:


```python
dfb = dfb.fillna(0).astype(int)
```

We display the result:


```python
dfb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>aa</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abhor</th>
      <th>ability</th>
      <th>able</th>
      <th>aboard</th>
      <th>abolish</th>
      <th>abolishment</th>
      <th>...</th>
      <th>wrestle</th>
      <th>wristband</th>
      <th>writhen</th>
      <th>wrongful</th>
      <th>yah</th>
      <th>yell</th>
      <th>yote</th>
      <th>youthfulness</th>
      <th>zipping</th>
      <th>zoom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9561 columns</p>
</div>



The dataset is now one hot encoded!

This will allow us to count the number of occurrences of each word more easily:


```python
words_sum = dfb.sum(axis = 0).T
```


```python
words_sum.head()
```




    a            2
    aa           3
    abandon      3
    abandoned    6
    abhor        1
    dtype: int64



The `words_sum` DataFrame contains the number of times each word appears.

**It is now sorted alphabetically from A to Z but I’d like to analyze the words that appear the most often.**

So let’s sort `words_sum` by decreasing order of occurrence:


```python
words_sum = words_sum.sort_values(ascending=False)
```

Now let’s display the words that appear most often in our dataset:


```python
words_sum_max = words_sum.head(20)

plt.figure(figsize=(16,5))
words_sum_max.plot.bar(rot=25)
```




    <AxesSubplot:>




![png](Readme_files/Readme_68_1.png)


These words refer to school and elections. At this point, we can’t say much about these words.

It is interesting to display them now in order to compare them later during the Multivariate Analysis.

Indeed, the most frequent words in the global dataset may not be the same as in the `Effective` discourse.

## **Multivariate Analysis**

We now have a much more accurate view of our dataset by analyzing:

- Effectiveness (target)
- Discourse types
- Length of the discourses
- Words that make up discourses

Let’s move on to the Multivariate Analysis.

Multivariate Analysis is the examination of our features by putting them in relation with our target.

This will allow us to make hypotheses about the dataset.

> Here we are in the theorization phase.

**The question associated with Multivariate Analysis is: Is there a link between our features and the target?**

### **Categorical Data**

First, let’s start with the categorical data.

Is there a relationship between discourse types (`discourse_type`) and their effectiveness (`discourse_effectiveness`)?

We display the number of occurrences of each of these types as a function of effectiveness:


```python
import numpy as np

plt.figure(figsize=(8,8))
cross = pd.crosstab(index=df['discourse_effectiveness'],columns=df['discourse_type'],normalize='index')
cross.plot.barh(stacked=True,rot=40,cmap='plasma').legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('% distribution per category')
plt.xticks(np.arange(0,1.1,0.2))
plt.title("Forestfire damage each {}".format('discourse_type'))
plt.show()
```


    <Figure size 576x576 with 0 Axes>



![png](Readme_files/Readme_74_1.png)



```python
cross
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>discourse_type</th>
      <th>Claim</th>
      <th>Concluding Statement</th>
      <th>Counterclaim</th>
      <th>Evidence</th>
      <th>Lead</th>
      <th>Position</th>
      <th>Rebuttal</th>
    </tr>
    <tr>
      <th>discourse_effectiveness</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adequate</th>
      <td>0.338323</td>
      <td>0.092721</td>
      <td>0.054822</td>
      <td>0.289079</td>
      <td>0.059303</td>
      <td>0.132717</td>
      <td>0.033036</td>
    </tr>
    <tr>
      <th>Effective</th>
      <td>0.365108</td>
      <td>0.088462</td>
      <td>0.044821</td>
      <td>0.309350</td>
      <td>0.073236</td>
      <td>0.082565</td>
      <td>0.036457</td>
    </tr>
    <tr>
      <th>Ineffective</th>
      <td>0.228258</td>
      <td>0.089910</td>
      <td>0.031724</td>
      <td>0.488394</td>
      <td>0.056329</td>
      <td>0.072733</td>
      <td>0.032652</td>
    </tr>
  </tbody>
</table>
</div>



Impressive! At first glance, one can see that the more a discourse is of `Claim` type, the more `Effective` it is.

But is this really the case?

If we go back to our Univariate Analysis, we can see that `Claim` and `Evidence` are overrepresented types in our dataset. It is therefore logical to see them overrepresented in this analysis.

In fact, it would be more logical to evaluate this distribution in a statistical way. Thus all `discourse_type` would have the same weight in the dataset and the analysis would not be biased.

For example we have only 2.291 `Lead` against 12.105 `Evidence`. So there will be more `Evidence` in our analysis. This creates an imbalance. For `Effective`, we have 2,885 `Evidence` and 683 `Lead`. Does this mean that an `Evidence` discourse is more effective than a Lead discourse?

To clarify this, we need to perform a normalized analysis.



#### **Normalization**

To give the same weight to each `discourse_type` in the dataset, we need to normalize their occurrences.

We start by taking the number of occurrences of each `discourse_type` according to the `discourse_effectiveness`:


```python
cross_norm = pd.crosstab(index=df['discourse_effectiveness'],columns=df['discourse_type'])
```


```python
cross_norm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>discourse_type</th>
      <th>Claim</th>
      <th>Concluding Statement</th>
      <th>Counterclaim</th>
      <th>Evidence</th>
      <th>Lead</th>
      <th>Position</th>
      <th>Rebuttal</th>
    </tr>
    <tr>
      <th>discourse_effectiveness</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adequate</th>
      <td>7097</td>
      <td>1945</td>
      <td>1150</td>
      <td>6064</td>
      <td>1244</td>
      <td>2784</td>
      <td>693</td>
    </tr>
    <tr>
      <th>Effective</th>
      <td>3405</td>
      <td>825</td>
      <td>418</td>
      <td>2885</td>
      <td>683</td>
      <td>770</td>
      <td>340</td>
    </tr>
    <tr>
      <th>Ineffective</th>
      <td>1475</td>
      <td>581</td>
      <td>205</td>
      <td>3156</td>
      <td>364</td>
      <td>470</td>
      <td>211</td>
    </tr>
  </tbody>
</table>
</div>



Then we count the total number of occurrences of each `discourse_type`:


```python
count_type = df['discourse_type'].value_counts()
```

And finally we can normalize each of the occurrences according to the efficiency. We divide by the total number of occurrences of the type and multiple by the same number (1000):


```python
cross_norm['Claim'] = (cross_norm['Claim']/count_type['Claim'])*1000
cross_norm['Concluding Statement'] = (cross_norm['Concluding Statement']/count_type['Concluding Statement'])*1000
cross_norm['Counterclaim'] = (cross_norm['Counterclaim']/count_type['Counterclaim'])*1000
cross_norm['Evidence'] = (cross_norm['Evidence']/count_type['Evidence'])*1000
cross_norm['Lead'] = (cross_norm['Lead']/count_type['Lead'])*1000
cross_norm['Position'] = (cross_norm['Position']/count_type['Position'])*1000
cross_norm['Rebuttal'] = (cross_norm['Rebuttal']/count_type['Rebuttal'])*1000
```


```python
cross_norm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>discourse_type</th>
      <th>Claim</th>
      <th>Concluding Statement</th>
      <th>Counterclaim</th>
      <th>Evidence</th>
      <th>Lead</th>
      <th>Position</th>
      <th>Rebuttal</th>
    </tr>
    <tr>
      <th>discourse_effectiveness</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adequate</th>
      <td>592.552392</td>
      <td>580.423754</td>
      <td>648.618161</td>
      <td>500.950021</td>
      <td>542.994326</td>
      <td>691.848907</td>
      <td>557.073955</td>
    </tr>
    <tr>
      <th>Effective</th>
      <td>284.294899</td>
      <td>246.195166</td>
      <td>235.758601</td>
      <td>238.331268</td>
      <td>298.123090</td>
      <td>191.351889</td>
      <td>273.311897</td>
    </tr>
    <tr>
      <th>Ineffective</th>
      <td>123.152709</td>
      <td>173.381080</td>
      <td>115.623237</td>
      <td>260.718711</td>
      <td>158.882584</td>
      <td>116.799205</td>
      <td>169.614148</td>
    </tr>
  </tbody>
</table>
</div>



We now have normalized occurrences. All we need to do now is to create statistics.

For each efficiency, we sum the total number of normalized occurrences:


```python
cross_normSum = cross_norm.sum(axis=1)
print(cross_normSum)
```

    discourse_effectiveness
    Adequate       4114.461515
    Effective      1767.366810
    Ineffective    1118.171675
    dtype: float64


Then we use this sum to create our statistics:


```python
cross_norm.loc['Adequate'] = cross_norm.loc['Adequate']/cross_normSum['Adequate']
cross_norm.loc['Effective'] = cross_norm.loc['Effective']/cross_normSum['Effective']
cross_norm.loc['Ineffective'] = cross_norm.loc['Ineffective']/cross_normSum['Ineffective']
```


```python
cross_norm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>discourse_type</th>
      <th>Claim</th>
      <th>Concluding Statement</th>
      <th>Counterclaim</th>
      <th>Evidence</th>
      <th>Lead</th>
      <th>Position</th>
      <th>Rebuttal</th>
    </tr>
    <tr>
      <th>discourse_effectiveness</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adequate</th>
      <td>0.144017</td>
      <td>0.141069</td>
      <td>0.157644</td>
      <td>0.121753</td>
      <td>0.131972</td>
      <td>0.168151</td>
      <td>0.135394</td>
    </tr>
    <tr>
      <th>Effective</th>
      <td>0.160858</td>
      <td>0.139301</td>
      <td>0.133395</td>
      <td>0.134851</td>
      <td>0.168682</td>
      <td>0.108269</td>
      <td>0.154644</td>
    </tr>
    <tr>
      <th>Ineffective</th>
      <td>0.110138</td>
      <td>0.155058</td>
      <td>0.103404</td>
      <td>0.233165</td>
      <td>0.142091</td>
      <td>0.104456</td>
      <td>0.151689</td>
    </tr>
  </tbody>
</table>
</div>



When this is done, we display the normalized distribution of each of the types according to the efficiency:


```python
cross_norm.plot.barh(stacked=True,rot=40,cmap='inferno').legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('% distribution per category')
plt.xticks(np.arange(0,1.1,0.2))
plt.title("Forestfire damage each {}".format('discourse_type'))
plt.show()
```


![png](Readme_files/Readme_92_0.png)


This is much better and more logical!

Most types don’t seem to affect the effectiveness of the discourse.

**Most?**

**It seems that some stand out more than others.**

##### **Max occurence – discourse_type**

Let’s display the max values for each of the discourse_effectiveness:

For `Ineffective`:


```python
cross_norm.columns[(cross_norm == cross_norm.loc['Ineffective'].max()).any()].tolist()
```




    ['Evidence']



For `Adequate`:


```python
cross_norm.columns[(cross_norm == cross_norm.loc['Adequate'].max()).any()].tolist()
```




    ['Position']



For `Effective`:


```python
cross_norm.columns[(cross_norm == cross_norm.loc['Effective'].max()).any()].tolist()
```




    ['Lead']



In the Univariate Analysis, we saw that most of the long discourses are `Effective` and `Evidence` type.

We could have established a link between an `Evidence` type discourse and an `Effective` type discourse. But as we see here, the two are not correlated.

Indeed, most `Evidence` type discourses are `Ineffective`, while `Position` types are `Adequate` and `Lead` types are `Effective`.

Normalization is mandatory here to understand the data properly. If we had not done so, we would have come to the conclusion that `Claim` type discourse is the most useful for `Adequate` and `Effective` discourse. This is not true. This bias is due to the lack of data in our dataset (the lack of generalization).

Most of our data are of type `Evidence`(12,105) and `Claim`(11,977) while only 2,291 rows are counted for type `Lead`.

Moreover, another bias may remain. Do we have enough data on `Lead` to draw a conclusion on this label?

For now, the contest has only one dataset, so let’s assume we do 😉

### **NLP Data**

#### **Discourse Length**

Let’s continue on the analysis of discourse length in relation to effectiveness:


```python
plt.figure(figsize=(8,4))
sns.boxplot(data=df, x='discourse_length', y='discourse_effectiveness').set_xlim(-1, 220)
plt.show()
```


![png](Readme_files/Readme_105_0.png)


The longer a discourse is, the more likely it is to be `Effective`. We can also display the average number of words in each category.


```python
df.groupby(['discourse_effectiveness'])['discourse_length'].median()
```




    discourse_effectiveness
    Adequate       24.0
    Effective      39.0
    Ineffective    30.0
    Name: discourse_length, dtype: float64



#### **Words Analysis**

Finally, the analysis of words.

Do the words chosen have an impact on the effectiveness of a discourse? If so, we should find disparities in effectiveness.

We start by extracting the `discourse_effectiveness` column from the dataset and add it to `dfb`, the encoded one-hot discourses:


```python
dfb['discourse_effectiveness'] = df['discourse_effectiveness']
```


```python
dfb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>aa</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abhor</th>
      <th>ability</th>
      <th>able</th>
      <th>aboard</th>
      <th>abolish</th>
      <th>abolishment</th>
      <th>...</th>
      <th>wristband</th>
      <th>writhen</th>
      <th>wrongful</th>
      <th>yah</th>
      <th>yell</th>
      <th>yote</th>
      <th>youthfulness</th>
      <th>zipping</th>
      <th>zoom</th>
      <th>discourse_effectiveness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Adequate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Adequate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Adequate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Adequate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Adequate</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9562 columns</p>
</div>



We will separate this dataset into 3 DataFrame:

- `Effective`
- `Adequate`
- `Ineffective`

And analyze the words contained in each of them:


```python
dfb_ineffective = dfb.loc[dfb['discourse_effectiveness'] == 'Ineffective'].drop('discourse_effectiveness', axis=1)
dfb_adequate = dfb.loc[dfb['discourse_effectiveness'] == 'Adequate'].drop('discourse_effectiveness', axis=1)
dfb_effective = dfb.loc[dfb['discourse_effectiveness'] == 'Effective'].drop('discourse_effectiveness', axis=1)
```

As for the Univariate Analysis, we sum the occurrence of each word:


```python
words_sum_ineffective = dfb_ineffective.sum(axis = 0).T
words_sum_adequate = dfb_adequate.sum(axis = 0).T
words_sum_effective  = dfb_effective.sum(axis = 0).T
```

We sort them by descending order, the greatest number of occurrences first:




```python
words_sum_ineffective = words_sum_ineffective.sort_values(ascending=False)
words_sum_adequate = words_sum_adequate.sort_values(ascending=False)
words_sum_effective = words_sum_effective.sort_values(ascending=False)
```

And we take the first 20 occurrences:




```python
words_sum_ineffective_max = words_sum_ineffective.head(500)
words_sum_adequate_max = words_sum_adequate.head(500)
words_sum_effective_max = words_sum_effective.head(500)
```

We can display the graph for each of the DataFrame but here I prefer to group them in a single DataFrame and display the head.

This will allow us to compare more simply the occurrence of words according to the three types of effectiveness.


```python
pd.DataFrame(list(zip(list(words_sum_effective_max.index),
                      list(words_sum_adequate_max.index),
                      list(words_sum_ineffective_max.index))),
             columns =['Effective', 'Adequate', 'Ineffective']).head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Effective</th>
      <th>Adequate</th>
      <th>Ineffective</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>student</td>
      <td>student</td>
      <td>student</td>
    </tr>
    <tr>
      <th>1</th>
      <td>people</td>
      <td>people</td>
      <td>people</td>
    </tr>
    <tr>
      <th>2</th>
      <td>would</td>
      <td>would</td>
      <td>would</td>
    </tr>
    <tr>
      <th>3</th>
      <td>school</td>
      <td>vote</td>
      <td>vote</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vote</td>
      <td>school</td>
      <td>electoral</td>
    </tr>
    <tr>
      <th>5</th>
      <td>one</td>
      <td>electoral</td>
      <td>school</td>
    </tr>
    <tr>
      <th>6</th>
      <td>get</td>
      <td>college</td>
      <td>car</td>
    </tr>
    <tr>
      <th>7</th>
      <td>time</td>
      <td>car</td>
      <td>college</td>
    </tr>
    <tr>
      <th>8</th>
      <td>help</td>
      <td>like</td>
      <td>state</td>
    </tr>
    <tr>
      <th>9</th>
      <td>make</td>
      <td>get</td>
      <td>like</td>
    </tr>
    <tr>
      <th>10</th>
      <td>like</td>
      <td>state</td>
      <td>get</td>
    </tr>
    <tr>
      <th>11</th>
      <td>project</td>
      <td>could</td>
      <td>one</td>
    </tr>
    <tr>
      <th>12</th>
      <td>electoral</td>
      <td>one</td>
      <td>time</td>
    </tr>
    <tr>
      <th>13</th>
      <td>could</td>
      <td>time</td>
      <td>could</td>
    </tr>
    <tr>
      <th>14</th>
      <td>car</td>
      <td>think</td>
      <td>think</td>
    </tr>
    <tr>
      <th>15</th>
      <td>teacher</td>
      <td>make</td>
      <td>make</td>
    </tr>
    <tr>
      <th>16</th>
      <td>think</td>
      <td>help</td>
      <td>help</td>
    </tr>
    <tr>
      <th>17</th>
      <td>college</td>
      <td>president</td>
      <td>president</td>
    </tr>
    <tr>
      <th>18</th>
      <td>also</td>
      <td>also</td>
      <td>also</td>
    </tr>
    <tr>
      <th>19</th>
      <td>class</td>
      <td>way</td>
      <td>thing</td>
    </tr>
    <tr>
      <th>20</th>
      <td>state</td>
      <td>thing</td>
      <td>many</td>
    </tr>
    <tr>
      <th>21</th>
      <td>want</td>
      <td>many</td>
      <td>way</td>
    </tr>
    <tr>
      <th>22</th>
      <td>work</td>
      <td>dont</td>
      <td>dont</td>
    </tr>
    <tr>
      <th>23</th>
      <td>way</td>
      <td>want</td>
      <td>even</td>
    </tr>
    <tr>
      <th>24</th>
      <td>many</td>
      <td>even</td>
      <td>want</td>
    </tr>
    <tr>
      <th>25</th>
      <td>thing</td>
      <td>project</td>
      <td>know</td>
    </tr>
    <tr>
      <th>26</th>
      <td>know</td>
      <td>class</td>
      <td>project</td>
    </tr>
    <tr>
      <th>27</th>
      <td>president</td>
      <td>know</td>
      <td>face</td>
    </tr>
    <tr>
      <th>28</th>
      <td>dont</td>
      <td>need</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>29</th>
      <td>need</td>
      <td>work</td>
      <td>need</td>
    </tr>
  </tbody>
</table>
</div>



It does not seem that there is a remarkable difference. Neither between the Labels, nor between the Labels and the global dataset.

**In fact, usually the difference lies in the following lines. The first ones being always shared on the whole dataset.**

Therefore, I invite you to display on your side the 100 or 500 most frequent words according to the `discourse_effectiveness` and tell us in comments your analysis 🔥

From my personal studies, I know that verbs appear more in effective discourse. Is this true in our dataset? If so, it may be a useful clue for us.

#### **Tags Analysis**

To study the occurrence of verbs, we use [NLTK tags.](https://inside-machinelearning.com/en/nltk-quickly-know-the-tag-and-their-meanings/)

NLTK is an NLP library that can analyze sentences and extract tags.

**For example it can identify nouns, verbs, adjectives, etc.**

Perfect for us!

Let’s take our words and their occurrences. We’ll create a new `word` column thanks to the indexes:


```python
words_sum_ineffective = words_sum_ineffective.to_frame()
words_sum_ineffective['word'] = words_sum_ineffective.index

words_sum_adequate = words_sum_adequate.to_frame()
words_sum_adequate['word'] = words_sum_adequate.index

words_sum_effective = words_sum_effective.to_frame()
words_sum_effective['word'] = words_sum_effective.index
```

We now have a column representing the number of occurrences and a column representing the word.

Let’s add a last column representing the tag thanks to the `pos_tag` function of `nltk.tag`:


```python
from nltk import tag
nltk.download('averaged_perceptron_tagger')

words_sum_ineffective['tag'] = [x[1] for x in tag.pos_tag(words_sum_ineffective['word'])]
words_sum_adequate['tag'] = [x[1] for x in tag.pos_tag(words_sum_adequate['word'])]
words_sum_effective['tag'] = [x[1] for x in tag.pos_tag(words_sum_effective['word'])]
```

    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /Users/tomkeldenich/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!


Tags are very detailed. For example there are 4 types of verbs (past, present, etc) which all have a different tag. This detail is not interesting for us. Therefore we clean these tags to keep only the essential (one tag by verb, one tag by adjective etc):


```python
def easyTag(x):
    if x.startswith('VB'):
        x = 'VB'
    elif x.startswith('JJ'):
        x = 'JJ'
    elif x.startswith('RB'):
        x = 'RB'
    elif x.startswith('NN'):
        x = 'NN'
        
    return x

words_sum_ineffective['tag'] = words_sum_ineffective['tag'].apply(lambda x: easyTag(x))
words_sum_adequate['tag'] = words_sum_adequate['tag'].apply(lambda x: easyTag(x))
words_sum_effective['tag'] = words_sum_effective['tag'].apply(lambda x: easyTag(x))
```


```python
words_sum_ineffective.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>word</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>student</th>
      <td>2503</td>
      <td>student</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>people</th>
      <td>1865</td>
      <td>people</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>would</th>
      <td>1749</td>
      <td>would</td>
      <td>MD</td>
    </tr>
    <tr>
      <th>vote</th>
      <td>1532</td>
      <td>vote</td>
      <td>VB</td>
    </tr>
    <tr>
      <th>electoral</th>
      <td>1301</td>
      <td>electoral</td>
      <td>JJ</td>
    </tr>
    <tr>
      <th>school</th>
      <td>1258</td>
      <td>school</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>car</th>
      <td>1143</td>
      <td>car</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>college</th>
      <td>1135</td>
      <td>college</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>state</th>
      <td>1052</td>
      <td>state</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>like</th>
      <td>1042</td>
      <td>like</td>
      <td>IN</td>
    </tr>
  </tbody>
</table>
</div>




We now have a DataFrame with our tag column:

- NN for nouns
- VB for verbs
- JJ for adjectives
- RB for adverbs
- … (you can consult the rest of the Tags on [our article dedicated to the subject](https://inside-machinelearning.com/en/nltk-quickly-know-the-tag-and-their-meanings/))

Finally, we count the number of occurrences per Tag:


```python
def count_tag(words_sum):
    tag_count = []
    for x in words_sum['tag'].unique():
        tmp = []
        tmp.append(x)
        tmp.append(words_sum[words_sum['tag'] == x][0].sum())
        tag_count.append(tmp)
    return pd.DataFrame(tag_count, columns= ['tag','count'])

tag_ineffective = count_tag(words_sum_ineffective).sort_values(by=['count'], ascending=False)
tag_adequate = count_tag(words_sum_adequate).sort_values(by=['count'], ascending=False)
tag_effective = count_tag(words_sum_effective).sort_values(by=['count'], ascending=False)
```

And we can display the result:


```python
plt.figure(figsize=(6,8))
plt.subplot(3,1,1)
sns.barplot(x="tag", y="count", data=tag_ineffective.iloc[:6])
plt.title('Tag for Ineffective')
plt.subplot(3,1,2)
sns.barplot(x="tag", y="count", data=tag_adequate.iloc[:6])
plt.title('Tag for Adequate')
plt.subplot(3,1,3)
sns.barplot(x="tag", y="count", data=tag_effective.iloc[:6])
plt.title('Tag for Effective')
plt.tight_layout()
plt.show()
```


![png](Readme_files/Readme_134_0.png)


Nouns appear most often in all types of discourse. This seems logical because they compose the majority of the sentences. However, this gives us little information.

Let’s analyze the rest of the tags by omitting the nouns:




```python
plt.figure(figsize=(6,8))
plt.subplot(3,1,1)
sns.barplot(x="tag", y="count", data=tag_ineffective.iloc[1:6], color="#066b8b")
plt.title('Tag for Ineffective')
plt.subplot(3,1,2)
sns.barplot(x="tag", y="count", data=tag_adequate.iloc[1:6], color="#066b8b")
plt.title('Tag for Adequate')
plt.subplot(3,1,3)
sns.barplot(x="tag", y="count", data=tag_effective.iloc[1:6], color="#066b8b")
plt.title('Tag for Effective')
plt.tight_layout()
plt.show()
```


![png](Readme_files/Readme_136_0.png)


Here, the number of occurrences doesn’t matter. Since `Adequate` discourse has many more words, there will obviously be more words in each tag of this section.

Here we have to look at the tag ranking.

What we can see is that `Effective` discourse contains more verbs (VB) than the other types.

Our hypothesis seems to be confirmed!

Let’s go further by analyzing the number of verbs per speech according to the effectiveness.

#### **Average number of verbs by effectiveness**

This time we apply `pos_tag` on all our words preprocessed in `df_words`.

To recall this DataFrame contains one discourse per line with only the important words (without stopwords, special characters, etc). Using this DataFrame will facilitate tag counting:


```python
list_tags = []
for i in range(len(df_words)):
    list_tags.append([easyTag(x[1]) for x in tag.pos_tag(df_words[i])])
```

Then we count the number of verbs in each row:


```python
df_tag = pd.DataFrame(columns=['VB'])

for i in range(len(list_tags)):
    df_tag = df_tag.append({'VB': list_tags[i].count('VB')}, ignore_index=True)
```

We extract the `discourse_effectiveness` column and add it to df_tag:


```python
df_tag['discourse_effectiveness'] = df['discourse_effectiveness']
```


```python
df_tag.head()
```

Finally we display the average number of verbs per effectiveness.

For `Ineffective`:


```python
VB_ineffective = df_tag.loc[df_tag['discourse_effectiveness'] == 'Ineffective']
VB_ineffective['VB'].sum() / len(VB_ineffective)
```

For `Adequate`:


```python
VB_adequate = df_tag.loc[df_tag['discourse_effectiveness'] == 'Adequate']
VB_adequate['VB'].sum() / len(VB_adequate)
```

For `Effective`:


```python
VB_effective = df_tag.loc[df_tag['discourse_effectiveness'] == 'Effective']
VB_effective['VB'].sum() / len(VB_effective)
```

Dans un discours `Effective` on décompte 5 verbe moyens utilisés. C’est 2 de plus que dans un discours `Adequate` et 1 de plus que dans un discours `Ineffective`.

## **Conclusion**

To improve a discourse:
- Use about 39 words
- 5 verbs (but also adjectives)
- Have a Lead (a statistic, a quote, a description) in your discourse to grab the reader’s attention and direct them to the thesis.
- Avoid sharing ideas that support assertions, counter-affirmations, or rebuttals (most Evidence type discourses are ineffective).

To go further, there are many analyses that can be done. Questions that we haven’t answered. For example, we could study which words make up `Lead` vs. `Evidence` discourse.

Biases still remain, for example the average number of verbs is higher in `Effective` discourses but the average number of words is higher too. Is this a prerequisite to have an `Effective` discourse or is it simply a bias of the dataset?

We could analyze this dataset for days and list all the biases. But now that we have a first detailed and consistent analysis, the most important thing is to take action and use a Machine Learning model to achieve the main goal: classify the students’ discourses as “effective”, “adequate” or “ineffective”!

See you in a next post 😉


```python

```
