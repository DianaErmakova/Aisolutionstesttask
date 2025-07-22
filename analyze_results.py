import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from prepare_data import X
from simple_model import y_test, pred_logreg, logreg

cm = confusion_matrix(y_test, pred_logreg)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Fail', 'Predicted Pass'],
    yticklabels=['Actual Fail', 'Actual Pass']
)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок для логистической регрессии')
plt.show()

coefficients = logreg.coef_[0]
feature_names = X.columns
coef_df = pd.DataFrame({'Признак': feature_names, 'Коэффициент': coefficients})
coef_df['Абсолютное значение'] = coef_df['Коэффициент'].abs()
coef_df = coef_df.sort_values(by='Абсолютное значение', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Коэффициент', y='Признак', data=coef_df)
plt.title('Важные признаки по версии логистической регрессии')
plt.xlabel('Коэффициент')
plt.ylabel('Признак')
plt.grid(True)
plt.show()