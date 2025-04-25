import pandas as pd


def preprocess_input(form_data, encoders, model_type='linear'):
    """
    Preprocess user input for model prediction

    Args:
        form_data (dict): User input from the form
        encoders (dict): Encoders for categorical features with expected columns
        model_type (str): Which model to preprocess for: 'linear', 'lasso', 'xgboost'

    Returns:
        numpy.ndarray: Processed input ready for model prediction
    """
    try:
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([{
            'annee': form_data['annee'],
            'kilometrage': form_data['kilometrage'],
            'puissance_fiscale': form_data['puissance_fiscale'],
            'etat': form_data['etat'],
            'marque': form_data['marque'],
            'modele': form_data['modele'],
            'boite': form_data['boite'],
            'carburant': form_data['carburant']
        }])

        # One-hot encode the input data
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Select the right columns
        if model_type == 'linear':
            columns = encoders['linear_columns']
        elif model_type == 'lasso':
            columns = encoders['lasso_columns']
        elif model_type == 'xgboost':
            columns = encoders['xgboost_columns']
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Add missing columns with zeros
        for col in columns:
            if col not in input_encoded:
                input_encoded[col] = 0

        # Remove extra columns not used during training
        extra_cols = set(input_encoded.columns) - set(columns)
        if extra_cols:
            input_encoded = input_encoded.drop(columns=list(extra_cols))

        # Reorder to match training order
        input_encoded = input_encoded[columns]

        return input_encoded.values

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise ValueError(f"Preprocessing failed: {str(e)}")
