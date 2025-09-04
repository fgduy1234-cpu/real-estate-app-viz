import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, request, jsonify, render_template_string

# Инициализация Flask
app = Flask(__name__)

# Генерация синтетических данных (в реальности — загрузка из CSV или БД)
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'area': np.random.uniform(30, 150, n_samples),  # площадь в м²
        'floor': np.random.randint(1, 25, n_samples),   # этаж
        'total_floors': np.random.randint(5, 30, n_samples),
        'year_built': np.random.randint(1950, 2023, n_samples),
        'rooms': np.random.randint(1, 6, n_samples),
        'district': np.random.choice(['Центр', 'Север', 'Юг', 'Запад', 'Восток'], n_samples),
        'has_elevator': np.random.choice([0, 1], n_samples),
        'parking_distance': np.random.uniform(0, 2, n_samples),  # км до парковки
    }
    # Создаем цену как функцию от параметров + шум
    price = (
        data['area'] * 100_000 +
        data['floor'] * 10_000 -
        data['floor'] * 500 * (data['floor'] > data['total_floors'] * 0.8) +
        (2023 - data['year_built']) * (-800) +
        data['rooms'] * 200_000 +
        {'Центр': 1_500_000, 'Север': 500_000, 'Юг': 300_000, 'Запад': 700_000, 'Восток': 400_000}[data['district'][0]] * np.ones(n_samples) +
        data['has_elevator'] * 150_000 -
        data['parking_distance'] * 200_000 +
        np.random.normal(0, 100_000, n_samples)
    )
    data['price'] = np.maximum(price, 1_000_000)  # минимальная цена
    return pd.DataFrame(data)

# Загрузка данных и обучение модели
df = generate_sample_data()

# Кодируем категориальные признаки
le_district = LabelEncoder()
df['district_encoded'] = le_district.fit_transform(df['district'])

# Выбор признаков
features = ['area', 'floor', 'total_floors', 'year_built', 'rooms', 'district_encoded', 'has_elevator', 'parking_distance']
X = df[features]
y = df['price']

# Разделение и обучение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка модели
y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Модель обучена. MAE: {mae:,.0f} руб., R²: {r2:.3f}")

# HTML шаблон для веб-интерфейса
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Оценка недвижимости</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .form { max-width: 500px; margin: 0 auto; }
        input, select { width: 100%; padding: 8px; margin: 8px 0; }
        button { background: #007BFF; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .result { margin-top: 20px; font-size: 1.2em; text-align: center; }
    </style>
</head>
<body>
    <div class="form">
        <h2>Оценка стоимости недвижимости</h2>
        <form id="form">
            <input type="number" name="area" placeholder="Площадь (м²)" required>
            <input type="number" name="floor" placeholder="Этаж" required>
            <input type="number" name="total_floors" placeholder="Всего этажей" required>
            <input type="number" name="year_built" placeholder="Год постройки" required>
            <input type="number" name="rooms" placeholder="Количество комнат" required>
            <select name="district" required>
                <option value="">Выберите район</option>
                {% for district in districts %}
                <option value="{{ district }}">{{ district }}</option>
                {% endfor %}
            </select>
            <select name="has_elevator" required>
                <option value="">Есть ли лифт?</option>
                <option value="1">Да</option>
                <option value="0">Нет</option>
            </select>
            <input type="number" step="0.1" name="parking_distance" placeholder="Расстояние до парковки (км)" required>
            <button type="submit">Рассчитать стоимость</button>
        </form>
        <div class="result" id="result"></div>
    </div>

    <script>
        document.getElementById('form').onsubmit = async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const response = await fetch('/predict', {
                method: 'POST',
                body: new URLSearchParams(formData)
            });
            const result = await response.json();
            document.getElementById('result').innerHTML = 
                `Оценочная стоимость: <b>${result.price.toLocaleString()} руб.</b>`;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    districts = le_district.classes_
    return render_template_string(HTML_TEMPLATE, districts=districts)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем данные из формы
        input_data = {
            'area': float(request.form['area']),
            'floor': int(request.form['floor']),
            'total_floors': int(request.form['total_floors']),
            'year_built': int(request.form['year_built']),
            'rooms': int(request.form['rooms']),
            'district': request.form['district'],
            'has_elevator': int(request.form['has_elevator']),
            'parking_distance': float(request.form['parking_distance']),
        }

        # Кодируем район
        if input_data['district'] not in le_district.classes_:
            return jsonify({'error': 'Неизвестный район'}), 400

        input_data['district_encoded'] = le_district.transform([input_data['district']])[0]

        # Подготовка вектора признаков
        features_input = np.array([[
            input_data['area'],
            input_data['floor'],
            input_data['total_floors'],
            input_data['year_built'],
            input_data['rooms'],
            input_data['district_encoded'],
            input_data['has_elevator'],
            input_data['parking_distance']
        ]])

        # Предсказание
        predicted_price = model.predict(features_input)[0]
        predicted_price = round(predicted_price, -3)  # округление до тысяч

        return jsonify({
            'price': int(predicted_price),
            'currency': 'RUB'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Запуск сервера на http://127.0.0.1:5000")
    app.run(debug=True)