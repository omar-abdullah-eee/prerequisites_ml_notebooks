#ML Prerequisites - One Shot

> Everything you need to know before diving into Machine Learning — covered quickly, clearly, and with real code examples.

##A Note from Meeee

Run these codes for learning — tried to cover most of the topics in these notebooks. Matplotlib and Seaborn are not that mandatory for ML when you're just starting out, so don't stress over them for now. Focus on Python, NumPy, and Pandas first and you'll be in great shape! 🙌

This repository is designed for **absolute beginners** who want to get into Machine Learning but don't want to spend months studying prerequisites. All notebooks are written by **Omar Abdullah** and cover the essential Python tools used in every ML project.

---

##What's Inside

| Notebook | Topics Covered |

| `Basic Python`| Core Python fundamentals |
| `NumPy` | NumPy arrays, math, and matrix operations |
| `Pandas` | Data manipulation and analysis with Pandas |

---

##1. Python for ML

**File:** `python_for_ml_omar.ipynb`

A quick-fire walkthrough of the Python concepts that show up constantly in ML code.

### Topics Covered

**Variables & Control Flow**
```python
x = int(input("enter a number:"))
if x > 4:
    print("big")
else:
    print("small")
```

**For Loops**
```python
fruits = ["apple", "mango", "pineapple", "grapes", "dates"]
for i in fruits:
    print(i)
```

**Data Structures**

- **Tuple** — Ordered, immutable, allows duplicates. Use `()`.
```python
name = ("omar", "abdullah", "prince")
# Convert to list to edit, then convert back
name = list(name)
name[0] = "rakib"
name = tuple(name)
```

- **List** — Ordered, mutable, allows duplicates. Use `[]`.
```python
dept = ["eee", "cse", "me", "ce"]
print(type(dept))  # <class 'list'>
```

- **Set** — Unordered, no duplicates. Use `{}`.
```python
fruits = {"mango", "apple", "watermelon"}
fruits.add("berry")
fruits.remove("mango")
```

- **Dictionary** — Key-value pairs.
```python
student = {'name': 'omar', 'age': 26, 'dept': 'EEE'}
student.update({'name': 'abdullah'})
del student['dept']
```

**Functions**
```python
def square(x):
    return x ** 2

# Positional arguments
def greet(first, second):
    print("Full name:", first, second)

# Keyword arguments
greet(first="Omar", second="Abdullah")
```

**Time & DateTime**
```python
import time
import datetime

time.ctime()           # Current time as string
time.sleep(2)          # Delay execution by 2 seconds
datetime.datetime.now()  # Full timestamp
datetime.date.today()   # Date only
```

**NumPy (intro)**
```python
import numpy as np
x = np.array(["omar", "osama", "naznin"])
print(type(x))  # <class 'numpy.ndarray'>
```

**Timing your code**
```python
%timeit (j**4 for j in range(1, 10))       # Regular Python
%timeit np.arange(1, 10)**4                 # NumPy (much faster!)
```

---

##2. NumPy

**File:** `NumPy_by_omar.ipynb`

NumPy is the backbone of all numerical computing in Python. ML frameworks like TensorFlow and PyTorch are built on its concepts.

### Topics Covered

**Creating Arrays**
```python
import numpy as np

# From a list
a = np.array([1, 2, 3, 4])

# Zero, ones, empty arrays
np.zeros((3, 2))
np.ones((3, 2))
np.empty((3, 2))

# Range array
np.arange(1, 10)

# Evenly spaced values
np.linspace(1, 5, 9)
```

**Multi-dimensional Arrays**
```python
a2 = np.array([[1, 2, 3], [4, 5, 6]])     # 2D
a3 = np.array([[[1, 2, 3], [4, 5, 6]]])   # 3D
an = np.array([1, 3, 3], ndmin=7)          # nD
```

**Random Arrays**
```python
np.random.rand(3, 2)       # Float [0.0, 1.0)
np.random.randn(3, 2)      # Normal distribution
np.random.randint(1, 10, 5) # Random integers
```

**Data Types**
```python
a = np.array([1, 2, 3], dtype=np.float64)
v = x.astype(np.float16)   # Type conversion
```

**Arithmetic Operations**
```python
a = np.array([1, 2, 4, 2])
np.add(a, b)
np.subtract(a, b)
np.multiply(a, b)
np.divide(a, b)
np.power(a, 2)
np.sqrt(a)
```

**Trigonometric Functions**
```python
np.sin(x), np.cos(x), np.tan(x)
np.arcsin(x), np.arccos(x), np.arctan(x)
# All angles in radians
```

**Statistical Functions**
```python
np.max(x), np.min(x)
np.mean(x), np.std(x), np.var(x)
np.sum(x), np.median(x)
np.cumsum(x), np.corrcoef(x, y)
```

**Shape & Reshaping**
```python
c = np.arange(18)
d = c.reshape(3, 6)    # 1D → 2D
e = c.reshape(2, 3, 3) # 1D → 3D
oneD = e.reshape(-1)   # Back to 1D
```

**Indexing & Slicing**
```python
x = np.arange(-3, 6)
x[2:8]       # Elements from index 2 to 7
x[:6]        # From start to index 5
x[2:7:2]    # With step size

# 2D indexing
y[1, 2]      # Row 1, Column 2
```

**Array Operations**
```python
# Concatenation
np.concatenate((x, y))
np.concatenate((a, b), axis=0)  # Stack rows
np.concatenate((a, b), axis=1)  # Stack columns

# Stacking shortcuts
np.hstack((a, b))   # Horizontal
np.vstack((a, b))   # Vertical
np.dstack((a, b))   # Depth (3D)

# Splitting
np.array_split(x, 3)

# Search
np.where(x == 2)        # Indices where condition is True
np.searchsorted(arr, 35) # Where to insert to keep sorted

# Sorting
np.sort(x)
np.sort(y, axis=1)   # Sort each row
```

**Copy vs View**
```python
cpy = np.copy(x)   # Independent copy — changes don't affect original
vi  = x.view()     # Mirror — changes affect original
```

**Matrix Operations**
```python
x = np.matrix([[1, 2], [3, 4]])
np.transpose(x)           # Transpose
np.linalg.inv(x)          # Inverse
np.linalg.det(x)          # Determinant
np.linalg.matrix_power(x, 5)  # Power
```

---

## 🐼 3. Pandas

**File:** `pandas_by_omar.ipynb`

Pandas is your go-to tool for loading, cleaning, and exploring datasets before feeding them to any ML model.

### Topics Covered

**Loading Data**
```python
import pandas as pd

data = pd.read_csv("matches.csv")
data.head()   # First 5 rows
data.shape    # (rows, columns)
```

**Filtering / Masking**
```python
# Single condition
fetch = data['city'] == 'Kolkata'
data[fetch]

# Multiple conditions — use & instead of 'and'
mask1 = data['city'] == 'Indore'
mask2 = data['date'] >= '2012-01-01'
data[mask1 & mask2]
```

**Value Counts**
```python
data['venue'].value_counts()          # Most common values
data['winner'].value_counts().head()  # Top 5
```

**Plotting with Pandas**
```python
import matplotlib.pyplot as plt

data['winner'].value_counts().plot(kind='pie')
data['winner'].value_counts().plot(kind='bar')
data['winner'].value_counts().plot(kind='barh')  # Horizontal bar
data['win_by_wickets'].plot(kind='hist')          # Histogram
```

**Sorting**
```python
data.sort_values('season', ascending=False)

# Sort by multiple columns
data.sort_values(['season', 'city', 'win_by_runs'],
                  ascending=[True, False, False])
```

**drop_duplicates**
```python
# Get IPL winners per year (last match = final = winner)
data.drop_duplicates('season', keep='last') \
    .sort_values('season')[['season', 'winner']]
```

**GroupBy**
```python
ssn = data.groupby('season')
ssn.size()              # Matches per season
ssn.first()             # First row of each group
ssn.get_group(2011)     # Specific group

# Aggregation
ssn.sum(['win_by_runs', 'win_by_wickets'])
ssn.mean(['win_by_runs'])
ssn.max(['win_by_runs'])
```

**merge() — Combining DataFrames**
```python
merge_df = dl.merge(data, left_on='match_id', right_on='id')
```

**Pivot Table**
```python
# Rows = seasons, Columns = batsmen, Values = runs
ac = merge_df.pivot_table(
    index='season',
    columns='batsman',
    values='batsman_runs',
    aggfunc='sum'
)
```

**Handling Missing Data**
```python
df.isnull().any()    # Which columns have nulls?
df.isnull().sum()    # How many nulls per column?

df.dropna(axis=0)                        # Drop rows with any NaN
df.dropna(axis=1)                        # Drop columns with any NaN
df.dropna(axis=0, how='all')             # Drop only fully-empty rows
df.dropna(subset=['col1', 'col2'])       # Drop rows with NaN in specific cols

df['city'].fillna('Unknown')             # Fill with a value
df['col'].fillna(method='ffill')         # Forward fill
df['col'].fillna(method='bfill')         # Backward fill
```

**Other Useful Operations**
```python
data.corr(numeric_only=True)             # Correlation matrix
data.rename(columns={'id': 'match_no'})  # Rename columns
data.set_index('id')                     # Set custom index
data.reset_index()                       # Reset to default index
```

---

## 🛠️ How to Run These Notebooks

### Option A — Google Colab (Easiest, no setup needed)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload any `.ipynb` file from this repo
4. Click **Runtime → Run all**

### Option B — Locally with Jupyter
```bash
pip install jupyter numpy pandas matplotlib scikit-learn
jupyter notebook
```
Then open any `.ipynb` file from the browser interface.

---

##Requirements

```
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Install all at once:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

---

## 🗺️ Learning Path

```
Python Basics  →  NumPy  →  Pandas  →  You're ready for ML! 🎉
```

Once you're comfortable with these three notebooks, you're ready to start learning:
- **Scikit-learn** — classical ML algorithms
- **Matplotlib / Seaborn** — data visualization
- **TensorFlow / PyTorch** — deep learning

---

##Author

**Omar Abdullah**  
Notebooks written and maintained as part of a self-learning ML journey.

---

##Contributing

Found a bug or want to add more examples? Feel free to open an issue or submit a pull request. All contributions welcome!

This repo is **open for all** — share it with your friends as much as you can! The more people learn, the better. 🤝 If this helped you, consider giving it a ⭐ and passing it along to anyone starting their ML journey.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
