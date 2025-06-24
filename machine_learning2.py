import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data.csv')
x = df[['StudyHours','Attendance', 'ProjectsCompleted', 'InternshipMonths' ]]
y = df['GPA']

model = LinearRegression()
model.fit(x, y)

kate_cv = pd.DataFrame([{
    'StudyHours': 39,
    'Attendance': 90.42857143,
    'ProjectsCompleted': 3.857142857,
    'InternshipMonths': 2.571428571
}])


predicted_gpa = model.predict(kate_cv)[0]
print(f"Predicted GPA for Kate: {predicted_gpa:.2f}")

