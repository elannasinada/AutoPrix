import pandas as pd


def preprocess_input(form_data, encoders, model_type='linear'):
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

        print("[DEBUG] Raw input data (before encoding):")
        print(input_df)

        # One-hot encode the input data
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        print("[DEBUG] Encoded input data (after pd.get_dummies, before matching columns):")
        print(input_encoded)

        # Select the right columns for the model
        if model_type == 'linear':
            columns = encoders['linear_columns']
        elif model_type == 'lasso':
            columns = encoders['lasso_columns']
        elif model_type == 'xgboost':
            columns = encoders['xgboost_columns']
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"[DEBUG] Selected columns for {model_type} model: {columns}")

        # Add missing columns with zeros
        for col in columns:
            if col not in input_encoded:
                input_encoded[col] = 0

        print("[DEBUG] Final encoded input data (after adding missing columns with zeros):")
        print(input_encoded)

        # Remove extra columns not used during training
        extra_cols = set(input_encoded.columns) - set(columns)
        if extra_cols:
            input_encoded = input_encoded.drop(columns=list(extra_cols))

        print("[DEBUG] Final encoded input data (after removing extra columns):")
        print(input_encoded)

        # Reorder to match training order
        input_encoded = input_encoded[columns]

        print("[DEBUG] Final processed input data (after reordering columns):")
        print(input_encoded)

        return input_encoded.values

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise ValueError(f"Preprocessing failed: {str(e)}")
