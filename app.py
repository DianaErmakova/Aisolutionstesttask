import gradio as gr
import numpy as np

coef = np.array([-0.6, -0.4, 1.5])
intercept = -0.1


def predict(studytime, failures, prev_grades):
    X = np.array([studytime, failures, prev_grades])
    logit = intercept + np.dot(coef, X)
    prob = 1 / (1 + np.exp(-logit))
    pred_class = "Сдаст" if prob >= 0.5 else "Не сдаст"

    impacts = {
        "Время на учебу": coef[0],
        "Неудачи": coef[1],
        "Предыдущие оценки": coef[2]
    }

    explanation = "Влияние факторов на результат:\n"
    for factor, weight in impacts.items():
        sign = "положительное" if weight > 0 else "отрицательное"
        explanation += f"- {factor}: {sign} ({weight})\n"
    explanation += f"\nВероятность сдачи: {prob:.2f}\nПрогноз: {pred_class}"
    return pred_class, explanation


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 5, step=1, label="Время на учебу (study time)"),
        gr.Slider(0, 5, step=1, label="Количество неудач (failures)"),
        gr.Slider(0, 5, step=1, label="Предыдущие оценки (previous grades)")
    ],
    outputs=[
        gr.Textbox(label="Прогноз"),
        gr.Textbox(label="Объяснение")
    ],
    title="Прогноз успешности сдачи экзамена",
    description="Введите параметры, чтобы узнать вероятность сдачи и понять влияние факторов."
)

iface.launch()
