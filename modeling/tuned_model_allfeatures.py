import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TunedModel:
    def __init__(self, models, features, target, test_size=0.2, random_state=42, hyperparameter_tuning=False):
        self.models = models
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.hyperparameter_tuning = hyperparameter_tuning
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def _preprocess_data(self):
        X = self.features.drop(self.target, axis=1)
        y = self.features[self.target]
        # Новая строка: идентификация нечисловых столбцов
        non_numeric_cols = X.select_dtypes(include=['object']).columns
        # Новая строка: удаление нечисловых столбцов
        X = X.drop(non_numeric_cols, axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        # Feature scaling
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    def _evaluate_model(self, model, name, param_grid=None):
        if self.hyperparameter_tuning and param_grid is not None:
            grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=-1)
            grid_search.fit(self.X_train_scaled, self.y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model
        best_model.fit(self.X_train_scaled, self.y_train)
        y_pred = best_model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        result_df = pd.DataFrame([{
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        }])
        # Проверяем, пуст ли основной DataFrame
        if self.results_df.empty:
            self.results_df = result_df
        else:
            # Объединяем текущий результат с основным DataFrame
            self.results_df = pd.concat([self.results_df, result_df], ignore_index=True)

    def evaluate_models(self):
        self._preprocess_data()
        # Создаем DataFrame для хранения результатов сравнения моделей
        self.results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        # Обучаем и оцениваем каждую модель
        for name, (model, param_grid) in self.models.items():
            print(f"Evaluating {name}...")
            self._evaluate_model(model, name, param_grid)
        # Отображаем результаты
        print(self.results_df)
