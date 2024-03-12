```python
import pandas as pd
df = pd.read_csv('salaries.csv')
df.head()
inputs = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']
from sklearn.preprocessing import LabelEncoder #LabelEncoder is used to convert string data into numbers
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

inputs
#drop the columns that are not needed 
inputs_n = inputs.drop(['company','job','degree'],axis='columns')
inputs_n
target
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
model.score(inputs_n,target)
model.predict([[2,1,0]]) #google, sales, bachelors
model.predict([[2,1,1]])
model.predict([[2,1,2]])
model.predict([[0,1,2]])

model.predict([[2,0,2]])
model.predict([[0,0,0]])

model.predict([[0,0,1]])

model.predict([[0,1,0]])

model.predict([[0,1,1]])

```
