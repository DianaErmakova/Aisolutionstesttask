import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("student-mat.csv", sep=";")
df = df[['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']]
print('просмотр первых пяти строк:')
print(df.head())

print('\nинформация о типах данных и пропущенных значениях:')
print(df.info())

print('\nпроверка на пропуски:')
print(df.isnull().sum())

print('\nнемного описательной статистики:')
print(df.describe())

print('\nразмер таблицы (строки, столбцы):')
print(df.shape)

print('\nсписок всех колонок:')
print(df.columns)

print('\nтипы данных по колонкам:')
print(df.dtypes)

print('\nколичество уникальных значений в каждой колонке:')
print(df.nunique())

print('\nколичество дублирующихся строк:')
print(df.duplicated().sum())

df['pass_exam'] = (df['G3'] >= 10).astype(int)
df = df.drop(columns=['G3'])

plt.figure(figsize=(6, 4))
df['absences'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title("Распределение пропусков занятий")
plt.xlabel("Количество пропусков")
plt.ylabel("Количество студентов")
plt.grid(False)
plt.show()

# Точечная диаграмма + линия регрессии
plt.figure(figsize=(6, 4))
sns.regplot(
    x='G1', y='G2', data=df,
    scatter_kws={'alpha':0.5},
    line_kws={'color':'red'}
)
plt.title("Корреляция между G1 и G2")
plt.xlabel("Оценка за первый период (G1)")
plt.ylabel("Оценка за второй период (G2)")
plt.grid(True)
plt.show()

correlation = df['G1'].corr(df['G2'])
print(f"Коэффициент корреляции между G1 и G2: {correlation:.2f}")

X = df.drop(columns=['pass_exam'])
y = df['pass_exam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)