import os

import numpy as np
import pandas as pd

from emutils import PACKAGE_DATA_FOLDER
from emutils.utils import attrdict

from ..kaggle import kaggle_dataset


def load_lendingclub(
    base_path=PACKAGE_DATA_FOLDER,
    directory='lendingclub',
    cleaning_type='ax',
    random_state=2020,
):

    random_state = np.random.RandomState(random_state)

    def target_clean(df):
        # Non-completed loans
        df = df[df['loan_status'] != 'Current']
        df = df[df['loan_status'] != 'In Grace Period']

        # The taget must not be NaN
        df = df.dropna(how='any', subset=['loan_status'])

        # Recode targets
        df['loan_status'] = df.loan_status.map({
            'Fully Paid': 0,
            'Charged Off': 1,
            'Late (31-120 days)': 1,
            'Late (16-30 days)': 1,
            'Does not meet the credit policy. Status:Fully Paid': 0,
            'Does not meet the credit policy. Status:Charged Off': 1,
            'Default': 1
        })
        return df.reset_index(drop=True).copy()

    def basic_cleaning(df):
        # Drop columns with NaN more than 90%
        drop_cols = df.columns[df.isnull().mean() > 0.9]
        df = df.drop(drop_cols, axis=1)

        # Drop records with more than 50% of NaN features
        df = df[(df.isnull().mean(axis=1) < .5)]

        df['verification_status'] = df.verification_status.map({'Verified': 0, 'Source Verified': 1, 'Not Verified': 2})
        df['debt_settlement_flag'] = df.debt_settlement_flag.map({'N': 0, 'Y': 1})
        df['initial_list_status'] = df.initial_list_status.map({'w': 0, 'f': 1})
        df['application_type'] = df.application_type.map({'Individual': 0, 'Joint App': 1})
        df['hardship_flag'] = df.hardship_flag.map({'N': 0, 'Y': 1})
        df['pymnt_plan'] = df.pymnt_plan.map({'n': 0, 'y': 1})
        df['disbursement_method'] = df.disbursement_method.map({'Cash': 0, 'DirectPay': 1})
        df['term'] = df.term.map({' 36 months': 0, ' 60 months': 1})
        df['grade'] = df['grade'].map({v: i for i, v in enumerate(np.sort(df['grade'].unique()))})
        df['sub_grade'] = df['sub_grade'].map({v: i for i, v in enumerate(np.sort(df['sub_grade'].unique()))})
        df['emp_length'] = df['emp_length'].apply(lambda x: x.replace('year', '').replace('s', '').replace('+', '').
                                                  replace('< 1', '0') if isinstance(x, str) else '-1').astype(int)
        df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x: int(x[-4:]))
        df['issue_d'] = pd.to_datetime(df['issue_d'])

        # Get rid of few customers with no home
        df = df[df['home_ownership'].apply(lambda x: x in ['OWN', 'RENT', 'MORTGAGE'])]
        df['home_ownership'] = df.home_ownership.map({'MORTGAGE': 0, 'OWN': 1, 'RENT': 2})

        return df.reset_index(drop=True).copy()

    def ax_cleaning(df):
        COLUMNS = ['loan_status', 'issue_d'] + sorted([
            'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
            'annual_inc', 'verification_status', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
            'revol_util', 'total_acc', 'application_type', 'mort_acc', 'pub_rec_bankruptcies'
        ])

        feature = 'dti'
        filtercol = (df[feature] < 0)
        df[feature][filtercol] = random_state.normal(24.5, .5, size=int((filtercol).sum()))
        filtercol = (df[feature].isnull())
        df[feature][filtercol] = -1

        feature = 'pub_rec_bankruptcies'
        filtercol = (df[feature].isnull())
        df[feature][filtercol] = df[feature].median()

        feature = 'mort_acc'
        filtercol = (df[feature].isnull())
        df[feature][filtercol] = 52

        feature = 'revol_util'
        filtercol = (df[feature].isnull())
        df[feature][filtercol] = random_state.normal(82.5, 3, size=int((filtercol).sum()))

        return df[COLUMNS].reset_index(drop=True).copy()

    dataset_location = kaggle_dataset('lendingclub', directory=directory, base_path=base_path)

    # Could not load the data (must be downloaded from Kaggle first)
    if dataset_location is None:
        return None

    else:
        df = pd.read_csv(os.path.join(dataset_location, 'accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv'))
        df = target_clean(df)
        df = basic_cleaning(df)

        if cleaning_type == 'ax':
            df = ax_cleaning(df)
        else:
            raise ValueError('Invalid cleaning type specified.')

    return attrdict(data=df.reset_index(drop=True),
                    class_names=['Good', 'Bad'],
                    target_name='loan_status',
                    split_date='issue_d')