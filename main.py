import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_collection_and_processing.mining_data import EthereumTransactionAnalyzer
from data_collection_and_processing.combining_data import DataCombiner
from data_collection_and_processing.exploratory_data_analysis import DataAnalyzer
from data_collection_and_processing.feature_selection import FeatureSelector

from modeling.tuned_model_allfeatures import TunedModel
from modeling.evaluation import AUPRCPlotter
from dotenv import load_dotenv

#EDA
kaggle_data_path = r'Data/transaction_dataset.csv'
combined_data_path = r'Data/address_data_combined.csv'
output_path = r'Data/new_combined_data.csv'
address_file_path = r'Data/for_model_addresses_mined_not_in_kaggle.csv'
data_path = 'Data/increased_transaction_dataset.csv'
output_file_path = r'Data/new_address_data_ethereum.csv'

analyzer = DataAnalyzer(kaggle_data_path)
analyzer.load_data()
analyzer.execute()

#Mining Data
load_dotenv()
api_key = ""
analyzer = EthereumTransactionAnalyzer(output_file_path)
analyzer.process_addresses()
combiner = DataCombiner(output_file_path,
                        kaggle_data_path,
                        output_path)
combiner.execute()
df = pd.read_csv(data_path)
feature_selector = FeatureSelector(df)
feature_selector.fit_model()

# Get important features
feature_selector.get_feature_importances()
feature_selector.select_top_features()
selected_df = feature_selector.get_selected_dataframe()

# Models with hyperparameter grids
models_dict = {
    'RandomForest': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    'KNeighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
}
# Hyperparameter Tuned with all features(had better results)
model_evaluator = TunedModel(models=models_dict, features=df, target='FLAG', hyperparameter_tuning=True)
model_evaluator.evaluate_models()

# Plotting the  curve
auprc_plotter = AUPRCPlotter(model_evaluator)
auprc_plotter.plot_auprc()