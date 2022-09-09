import os
import numpy as np
import pandas as pd

from emutils import PACKAGE_DATA_FOLDER
from emutils.utils import attrdict


def load_heloc(base_path=PACKAGE_DATA_FOLDER,
               directory='heloc',
               filename='HELOC.csv',
               random_state=2020,
               cleaning_type='default_clean'):
    """"
    Description of the dataset can be found here: http://dukedatasciencefico.cs.duke.edu/models/
                         
    """
    random_state = np.random.RandomState(random_state)

    def clean_heloc(data):
        data['RiskPerformance'] = data['RiskPerformance'].apply(lambda x: 1 if 'Bad' in x else 0)
        all_special_mask = np.all(
            ((data.drop(columns=['RiskPerformance']) <= -7) & (data.drop(columns=['RiskPerformance']) >= -9)).values,
            axis=1)
        data = data[~all_special_mask]
        data = data[np.sort(data.columns.values)]
        data['ExternalRiskEstimate'][data['ExternalRiskEstimate'] < 0] = data['ExternalRiskEstimate'].median()
        data['MSinceMostRecentDelq'][data['MSinceMostRecentDelq'] < 0] = \
            random_state.uniform(30, 80, int((data['MSinceMostRecentDelq'] < 0).sum())).astype(int)

        data['MSinceMostRecentInqexcl7days'][data['MSinceMostRecentInqexcl7days'] == -7] = -1
        data['MSinceMostRecentInqexcl7days'][data['MSinceMostRecentInqexcl7days'] == -8] = 25
        data['MSinceOldestTradeOpen'][data['MSinceOldestTradeOpen'] == -8] = random_state.uniform(
            145, 165, int((data['MSinceOldestTradeOpen'] == -8).sum())).astype(int)
        data['NetFractionInstallBurden'][data['NetFractionInstallBurden'] == -8] = random_state.normal(
            data['NetFractionInstallBurden'].median(), 7.5, size=int(
                (data['NetFractionInstallBurden'] == -8).sum())).astype(int)
        data['NetFractionRevolvingBurden'][data['NetFractionRevolvingBurden'] == -8] = random_state.normal(
            75, 5, size=int((data['NetFractionRevolvingBurden'] == -8).sum())).astype(int)
        data['NumInstallTradesWBalance'][data['NumInstallTradesWBalance'] == -8] = 0
        data['NumRevolvingTradesWBalance'][data['NumRevolvingTradesWBalance'] == -8] = random_state.normal(
            13, 1, size=int((data['NumRevolvingTradesWBalance'] == -8).sum())).astype(int)
        data['NumBank2NatlTradesWHighUtilization'][data['NumBank2NatlTradesWHighUtilization'] == -8] = 20
        data['PercentTradesWBalance'][data['PercentTradesWBalance'] == -8] = data['PercentTradesWBalance'].median()

        return data.reset_index(drop=True).copy()

    df = pd.read_csv(os.path.join(base_path, directory, filename))
    if 'default_clean' in cleaning_type:
        df = clean_heloc(df)

    return attrdict(
        data=df,
        target_name='RiskPerformance',
        class_names=['Good', 'Bad'],
    )