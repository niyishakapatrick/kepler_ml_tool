import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
import io
import altair as alt
import gc # Import garbage collector for memory management

# --- MEMORY OPTIMIZATION FUNCTION ---
def reduce_mem_usage(df):
    """Iterate through all the columns of a dataframe and modify data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    st.info(f"Initial memory usage of DataFrame: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # Use float32 for most floats to save memory
                df[col] = df[col].astype(np.float32)
        elif col_type == object and df[col].nunique() < len(df) * 0.5:
             # Convert low cardinality object columns to 'category'
             df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    st.success(f"Memory usage after optimization: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    gc.collect() # Force garbage collection
    return df
# -----------------------------------------------

# --- Streamlit Setup & Sidebar Configuration ---
st.set_page_config(page_title="ML Tabular Trainer", layout="wide")

# Sidebar for global settings
with st.sidebar:
    st.header("App Settings 🛠️")
    st.image("kepler_logo.png")
    # Toggle for "Extra large upload area"
    extra_large_upload = st.checkbox('Extra large upload area', value=True)
    # Global UI scale slider
    ui_scale = st.slider("Global UI Scale", 1.0, 2.0, 1.2, 0.1)
    
    # Customizable Accent colors
    accent_color_ui = st.color_picker('UI Title & Buttons Color (Coral)', '#FF7F50')
    accent_color_data = st.color_picker('Data & Task Section Color (Steel Blue)', '#4682B4')
    accent_color_analysis = st.color_picker('Model & Analysis Color (Sea Green)', '#3CB371')

    # Gradient Color Inputs
    st.subheader("Background Gradients")
    # UI/Upload Gradient (Light Coral to slightly lighter Coral)
    grad_start_ui = st.color_picker('Upload Start Color', '#00FFAA')
    grad_end_ui = st.color_picker('Upload End Color', '#F0F8FF')
    
    # Data Section Gradient (AliceBlue to a slightly darker shade)
    grad_start_data = st.color_picker('Data Start Color', '#F0F8FF')
    grad_end_data = st.color_picker('Data End Color', '#E0FFFF') # Azure
    
    # Analysis Section Gradient (Honeydew to a slightly darker shade)
    grad_start_analysis = st.color_picker('Analysis Start Color', '#F0FFF0')
    grad_end_analysis = st.color_picker('Analysis End Color', '#F5FFFA') # Mint Cream


    
    # Calculate scaled font sizes
    base_font_size = 14 * ui_scale
    title_font_size = 30 * ui_scale
    
    # Plot scale factor (Base size reduced to 2.0x2.0 inches for matplotlib plots)
    plot_scale = ui_scale / 1.0 

# --- Custom CSS ---
upload_css = f"""
    /* 1. Upload Dropzone Styling (UI Gradient) */
    .stFileUploader > div {{
        border: 4px dashed {accent_color_ui} !important;
        /* Applying linear gradient to the upload zone */
        background: linear-gradient(135deg, {grad_start_ui}, {grad_end_ui}) !important; 
        padding: {'50px' if extra_large_upload else '20px'} 20px !important;
        height: {'250px' if extra_large_upload else 'auto'} !important; /* Taller hit area */
        border-radius: 10px;
    }}
    .stFileUploader label {{
        font-size: {base_font_size}px !important;
        font-weight: bold;
    }}
"""

def get_card_css(color, start_color, end_color, name):
    return f"""
        .card-{name} {{
            /* Applying linear gradient to the content cards */
            background: linear-gradient(135deg, {start_color}, {end_color}); 
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }}
        .card-title-{name} {{
            font-size: {24 * ui_scale}px;
            font-weight: bold;
            color: {color};
            margin-bottom: 15px;
        }}
    """
    
# Robust CSS for Black background/White text for all dataframes within .data-info-style
data_info_style_css = f"""
    /* Target the overall container of st.dataframe */
    .data-info-style div[data-testid="stDataFrame"] {{
        background-color: black !important;
    }}
    /* Target the actual table and its cells within the container */
    .data-info-style div[data-testid="stDataFrame"] table,
    .data-info-style div[data-testid="stDataFrame"] th,
    .data-info-style div[data-testid="stDataFrame"] td,
    .data-info-style div[data-testid="stDataFrame"] div:first-child {{
        background-color: black !important;
        color: white !important;
    }}
    /* Target the text in the table cells */
    .data-info-style div[data-testid="stDataFrame"] span,
    .data-info-style div[data-testid="stDataFrame"] div {{
        color: white !important;
    }}
    /* Apply style to the general info text block */
    .data-info-style .stTextContainer p {{
        background-color: black;
        color: white;
        padding: 10px;
        border-radius: 5px;
        white-space: pre-wrap; /* Ensure text formatting is preserved */
    }}
"""

global_css = f"""
    <style>
        /* Apply global scale to various elements */
        html, body, [class*="st-"], .stFileUploader label, .stRadio label, .stSelectbox label, .stSlider label {{
            font-size: {base_font_size}px;
        }}
        .center-title {{
            text-align: center;
            font-size: {title_font_size}px;
            font-weight: bold;
            color: {accent_color_ui};
            margin-bottom: 25px;
        }}
        {upload_css}
        /* Custom card styling based on section color (NOW USING GRADIENTS) */
        {get_card_css(accent_color_data, grad_start_data, grad_end_data, 'data')}
        {get_card_css(accent_color_analysis, grad_start_analysis, grad_end_analysis, 'analysis')}
        {data_info_style_css}
    </style>
"""

st.markdown(global_css, unsafe_allow_html=True)


st.markdown("<div class='center-title'>📊 ML Tabular Data Trainer</div>", unsafe_allow_html=True)

# --- Upload data (Uses UI Color for Upload Zone) ---
uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Ensure pandas reads dates correctly, often 'infer_datetime_format=True' helps
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, infer_datetime_format=True)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    # --- APPLY MEMORY OPTIMIZATION HERE ---
    df = reduce_mem_usage(df)
    # --------------------------------------

    # Dataset Preview (Data Section Color & Background)
    # APPLIED 'data-info-style' class
    st.markdown(f"<div class='card-data data-info-style'><div class='card-title-data'>🔎 Dataset Preview</div>", unsafe_allow_html=True)
    # Catch for serialization error in preview
    try:
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not display DataFrame preview normally due to serialization issues. Falling back to plain text. Error: {e}")
        st.text(df.head().to_string())
        
    st.markdown("</div>", unsafe_allow_html=True)

    # Data Info (Data Section Color & Background)
    # APPLIED 'data-info-style' class
    st.markdown(f"<div class='card-data data-info-style'><div class='card-title-data'>📑 Data Info</div>", unsafe_allow_html=True)
    
    col_missing, col_types = st.columns(2)
    
    with col_missing:
        st.subheader("Missing Values")
        missing_df = df.isnull().sum().rename("Missing Count").to_frame()
        # Styled via CSS class 'data-info-style'
        try:
            st.dataframe(missing_df)
        except Exception as e:
            st.warning(f"⚠️ Could not display Missing Values DataFrame normally due to serialization issues. Falling back to plain text. Error: {e}")
            st.text(missing_df.to_string())
    
    with col_types:
        st.subheader("Data Types & Non-Null Counts")
        
        # Prepare Data Types and Non-Null Counts in a single DataFrame
        data_info_df = pd.DataFrame({
            'Dtype': df.dtypes,
            'Non-Null Count': df.count()
        }).reset_index().rename(columns={'index': 'Column'})
        
        # Styled via CSS class 'data-info-style'
        try:
            # Explicitly convert the Dtype column to string before showing (primary fix)
            data_info_df['Dtype'] = data_info_df['Dtype'].astype(str)
            st.dataframe(data_info_df, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not display Data Types DataFrame normally due to serialization issues. Falling back to plain text. Error: {e}")
            st.text(data_info_df.to_string())
        
    # Display overall memory usage and index info below the columns
    st.subheader("General Info:")
    buffer = io.StringIO()
    # The info() output is still useful for RangeIndex and Memory
    df.info(buf=buffer)
    s = buffer.getvalue()
    
    # --- REVISED LOGIC FOR CLEAN GENERAL INFO OUTPUT (Human-readable, no brackets) ---
    
    # 1. Get the Dtype counts as a clean string/list
    # Note: Use df.dtypes.to_frame() to retain index info for .value_counts() on dtypes
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    
    dtype_summary = "Dtype counts (Total Columns: " + str(df.shape[1]) + "):"
    for dtype, count in dtype_counts.items():
        # Ensure the output is clean
        dtype_summary += f"\n- {dtype.replace('(', '').replace(')', '').replace('[ns]', ' ')}: {count} columns"
        
    # 2. Extract RangeIndex and Memory Usage from df.info() output (s)
    # Get the index info (e.g., RangeIndex: 2000 entries, 0 to 1999)
    # Clean the index line by removing brackets
    try:
        range_index_line = [line for line in s.split('\n') if 'Index' in line][0].strip().replace('<', '').replace('>', '')
    except IndexError:
        range_index_line = "Index Info: Not Available"

    # Get the memory usage line
    try:
        memory_usage_line = [line for line in s.split('\n') if 'memory usage' in line][0].strip()
    except IndexError:
        memory_usage_line = "Memory Usage: Not Available"
    
    # 3. Assemble the final clean output
    clean_s = f"DataFrame Info Summary:\n"
    clean_s += f"{range_index_line}\n"
    clean_s += "\n"
    clean_s += f"{dtype_summary}\n"
    clean_s += f"{memory_usage_line}\n"

    st.text(clean_s) 
    
    st.markdown("</div>", unsafe_allow_html=True)
    # ---------------------------------------------------------------------------------

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Task Selection + Target Distribution (Data Section Color & Background) ---
    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown(f"<div class='card-data'><div class='card-title-data'>⚙️ Task Selection</div>", unsafe_allow_html=True)
        task_type = st.radio("Choose task type", ["Auto-detect", "Regression", "Classification"], index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    if task_type == "Auto-detect":
        if any(df[col].dtype == "O" or df[col].nunique() <= 10 for col in df.columns):
            task = "Classification"
            possible_targets = [col for col in df.columns if (df[col].dtype == "O") or (df[col].nunique() <= 10)]
        else:
            task = "Regression"
            possible_targets = numeric_cols
    elif task_type == "Regression":
        task = "Regression"
        possible_targets = numeric_cols
    else:
        task = "Classification"
        possible_targets = [col for col in df.columns if (df[col].dtype == "O") or (df[col].nunique() <= 10)]

    if not possible_targets:
        st.error(f"⚠️ No suitable target columns found for {task}. Please check your dataset.")
        st.stop()

    target_col = st.selectbox("🎯 Select target column (y)", possible_targets)
    
    # --- CLASSIFICATION LIMIT CHECK ---
    if task == "Classification":
        unique_classes = df[target_col].dropna().nunique()
        if unique_classes > 10:
            st.warning(f"⚠️ Target column '{target_col}' has **{unique_classes} unique classes**. Classification tasks with more than 10 classes are extremely slow and memory-intensive. Task auto-switched to **Regression** to prevent crashing.")
            task = "Regression"
            # If the original target was numeric but the user explicitly selected 'Classification',
            # we need to re-select from numeric targets upon switch.
            if target_col not in numeric_cols:
                # If the auto-switched column isn't numeric, find the first numeric one or stop.
                if numeric_cols:
                    target_col = st.selectbox("🎯 Select target column (y)", numeric_cols, key='reg_switch')
                else:
                    st.error("⚠️ Cannot switch to Regression: No numeric columns available.")
                    st.stop()
            
        
    st.info(f"📌 Task selected: **{task}**, Target: **{target_col}**")
    # ------------------------------------

    # FIX: Target Variable NaN handling
    if df[target_col].isnull().any():
        initial_rows = df.shape[0]
        df.dropna(subset=[target_col], inplace=True)
        st.warning(f"⚠️ Removed {initial_rows - df.shape[0]} rows where the target column ('{target_col}') was missing.")

    # Plot size reduced to 2.0x2.0 inches for Matplotlib consistency
    plot_figsize = (2.0 * plot_scale, 2.0 * plot_scale) 
    plot_font_size = 14 * ui_scale

    with col2:
        st.markdown(f"<div class='card-data'><div class='card-title-data'>📊 Target Distribution</div>", unsafe_allow_html=True)
        
        if task == "Classification":
            # --- Streamlit Plot using Altair ---
            # Plot only the top 10 classes if there are too many (for visual clarity)
            plot_df = df[target_col].value_counts().reset_index()
            plot_df.columns = ['Class', 'Count']
            plot_df = plot_df.head(10) # Limit for plotting speed and clarity
            
            chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('Class:N', sort='-y', axis=alt.Axis(title=target_col, labelFontSize=plot_font_size*0.7, titleFontSize=plot_font_size)),
                y=alt.Y('Count:Q', axis=alt.Axis(title='Count', labelFontSize=plot_font_size*0.7, titleFontSize=plot_font_size)),
                color=alt.value(accent_color_data),
                tooltip=['Class', 'Count']
            ).properties(
                title=alt.TitleParams("Class Distribution (Top 10)", fontSize=plot_font_size, anchor='middle'),
                # Set approximate pixel size based on scale for consistency
                width=300 * plot_scale, 
                height=300 * plot_scale
            ).interactive()

            st.altair_chart(chart, use_container_width=False)
            
        else:
            # --- Matplotlib Plot (Regression) ---
            fig, ax = plt.subplots(figsize=plot_figsize) 
            ax.hist(df[target_col].dropna(), bins=20, color=accent_color_data, edgecolor="black") 
            ax.set_title("Target Histogram", fontsize=plot_font_size, fontweight="bold") 
            ax.tick_params(axis='x', labelsize=plot_font_size * 0.7)
            ax.tick_params(axis='y', labelsize=plot_font_size * 0.7)
            st.pyplot(fig) # Call st.pyplot here
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Features / Labels
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Release original DataFrame to free up memory
    del df
    gc.collect()

    # **FIX** : Handle Datetime columns which cause model training errors
    datetime_cols = X.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns.tolist()
    if datetime_cols:
        X = X.drop(columns=datetime_cols)
        st.warning(f"⚠️ Dropped {len(datetime_cols)} datetime columns ({', '.join(datetime_cols)}) as they cannot be used directly in the current models. Consider extracting features like Year/Month/Day if needed.")
    # **END FIX**
    
    if task == "Classification":
        if y.dtype == "O" or y.dtype.name == "category":
            le = LabelEncoder()
            y = le.fit_transform(y)

    # --- Feature Encoding + Train/Test Split (Data Section Color & Background) ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div class='card-data'><div class='card-title-data'>🛠️ Feature Encoding Options</div>", unsafe_allow_html=True)
        # Recalculate categorical_cols AFTER dropping datetime columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist() 
        # --- NEW DEFAULT: Drop categorical columns (Index 1) ---
        encoding_choice = st.radio("How should categorical columns be handled?",
                                   ["One-hot encode (Warning: Memory Intensive)", "Drop categorical columns (Default)"], index=1)
        
        if encoding_choice.startswith("One-hot encode"): # Checks for the new string
            before_cols = X.shape[1]
            X = pd.get_dummies(X, drop_first=True)
            after_cols = X.shape[1]
            st.success(f"✅ {len(cat_cols)} categorical columns encoded. Added {after_cols-before_cols} features.")
        else: # "Drop categorical columns (Default)"
            X = X.drop(columns=cat_cols)
            st.warning(f"⚠️ {len(cat_cols)} categorical columns dropped. Remaining features: {X.shape[1]}")
            
        # FIX: Feature Imputation (Simple fill NA with mean/mode)
        # Handle NA in features AFTER encoding/dropping. This is a pragmatic fix.
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col].fillna(X[col].mean(), inplace=True)
                    st.info(f"ℹ️ Imputed missing values in feature **{col}** with its **mean**.")
                else:
                    # After get_dummies, this is unlikely, but for safety:
                    X[col].fillna(X[col].mode()[0], inplace=True)
                    st.info(f"ℹ️ Imputed missing values in feature **{col}** with its **mode**.")
        # FIX END
        gc.collect() # Force GC after feature engineering
        
        # --- Store feature names BEFORE deletion ---
        feature_names = X.columns.tolist()
        # -------------------------------------------
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='card-data'><div class='card-title-data'>🔀 Train/Test Split</div>", unsafe_allow_html=True)
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random seed", value=42)
        
        # Ensure that if the task is classification, stratification is possible
        if task == "Classification" and len(np.unique(y)) > 1 and all(y_val >= 2 for y_val in pd.Series(y).value_counts()):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            st.write("✅ Data split done with **stratification**")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            st.write("✅ Data split done (no stratification)")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Release original X and y dataframes immediately after split to free up memory ---
    del X, y 
    gc.collect() 
    # -----------------------------------------------------------------------------------------
    
    # Train models
    if task == "Regression":
        models = {
            "Random Forest": RandomForestRegressor(random_state=random_state),
            "XGBoost": xgb.XGBRegressor(random_state=random_state)
        }
    else:
        models = {
            "Random Forest": RandomForestClassifier(random_state=random_state),
            # Set enable_categorical=True just in case, though one-hot encoding should prevent categorical DTypes reaching here.
            "XGBoost": xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss", enable_categorical=True) 
        }

    results = {}
    
    # --- ADDED SPINNER FOR PROGRESS INDICATION ---
    with st.spinner(f"Training models for {task} task..."): 
        for name, model in models.items():
            try:
                # Training happens here. Target y should be clean now.
                model.fit(X_train, y_train)
                
                # Predict only once per model, using the retained X_test
                y_pred = model.predict(X_test)
                
                if task == "Regression":
                    results[name] = {"MSE": mean_squared_error(y_test, y_pred),
                                     "R2": r2_score(y_test, y_pred)}
                else:
                    results[name] = {"Accuracy": accuracy_score(y_test, y_pred)}

                # Save trained model 
                filename = f"{name.replace(' ','_')}_{task}.joblib"
                joblib.dump(model, filename)
                with open(filename, "rb") as f:
                    st.download_button(f"💾 Download {name} Model", f, file_name=filename) 
                
                # Force GC after each model to clean up internal objects
                gc.collect() 

            except Exception as e:
                st.error(f"⚠️ Model `{name}` failed: {str(e)}")
    
    # --- Release the large training data now, but keep X_test/y_test for evaluation ---
    del X_train, y_train
    gc.collect()

    # --- Extra Analysis (Analysis Section Color & Background) ---
    if task == "Classification" and results:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"<div class='card-analysis'><div class='card-title-analysis'>📋 Classification Report & Confusion Matrix</div>", unsafe_allow_html=True)
            chosen_model = st.selectbox("Choose model for evaluation", list(models.keys()))
            model = models[chosen_model]
            try:
                # X_test is available here
                y_pred = model.predict(X_test)
                
                # --- Classification Report Filtering Logic ---
                full_report = classification_report(y_test, y_pred, zero_division=0)
                num_classes = len(np.unique(y_test)) 
                REPORT_THRESHOLD = 50 

                st.text("Classification Report:")
                
                if num_classes > REPORT_THRESHOLD:
                    # Filter the report string to only show the summary lines
                    report_lines = full_report.split('\n')
                    # Keep header lines and the summary lines
                    summary_lines = [report_lines[0], report_lines[1], report_lines[2]] 
                    summary_lines.extend([line for line in report_lines if 'accuracy' in line or 'macro avg' in line or 'weighted avg' in line])
                    
                    st.warning(f"⚠️ Test set has {num_classes} unique classes. Showing only summary metrics to prevent excessive output.")
                    st.text('\n'.join(summary_lines))
                else:
                    # Print the full report if the number of classes is manageable
                    st.text(full_report)
                # --- END Filtering Logic ---

                # Apply scaled figsize and font size
                fig, ax = plt.subplots(figsize=plot_figsize)
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens", ax=ax, 
                            annot_kws={"size": plot_font_size * 0.5})
                ax.set_title("Confusion Matrix", fontsize=plot_font_size * 0.6, fontweight="bold")
                ax.set_xlabel("Predicted Label", fontsize=plot_font_size * 0.6)
                ax.set_ylabel("True Label", fontsize=plot_font_size * 0.6)
                ax.tick_params(axis='x', labelsize=plot_font_size * 0.4)
                ax.tick_params(axis='y', labelsize=plot_font_size * 0.4)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ Could not generate classification details: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='card-analysis'><div class='card-title-analysis'>🔍 Feature Importance</div>", unsafe_allow_html=True)
            chosen_model_imp = st.selectbox("Choose model for feature importance", list(models.keys()), key="importance")
            model = models[chosen_model_imp]
            try:
                importances = model.feature_importances_
                # --- Use the stored feature_names list (Fixes 'X' not defined) ---
                feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
                # ---------------------------------------------

                # Apply scaled figsize and font size
                fig, ax = plt.subplots(figsize=plot_figsize)
                sns.barplot(x="Importance", y="Feature", data=feat_imp.head(15), ax=ax, color=accent_color_analysis, edgecolor="black") 
                ax.set_title("Top 15 Features", fontsize=plot_font_size * 0.6, fontweight="bold")
                ax.set_xlabel("Importance", fontsize=plot_font_size * 0.6)
                ax.set_ylabel("Feature", fontsize=plot_font_size * 0.6)
                ax.tick_params(axis='x', labelsize=plot_font_size * 0.4)
                ax.tick_params(axis='y', labelsize=plot_font_size * 0.4)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ Could not generate feature importance: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # --- Delete test data now that analysis is complete ---
        del X_test, y_test 
        gc.collect() 
        # -----------------------------------------------------------

    elif task == "Regression" and results:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"<div class='card-analysis'><div class='card-title-analysis'>📈 Results</div>", unsafe_allow_html=True)
            try:
                st.dataframe(pd.DataFrame(results).T, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display Results DataFrame normally due to serialization issues. Falling back to plain text. Error: {e}")
                st.text(pd.DataFrame(results).T.to_string())
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='card-analysis'><div class='card-title-analysis'>🔍 Feature Importance</div>", unsafe_allow_html=True)
            chosen_model_imp = st.selectbox("Choose model for feature importance", list(models.keys()), key="importance_reg")
            model = models[chosen_model_imp]
            try:
                importances = model.feature_importances_
                # --- Use the stored feature_names list (Fixes 'X' not defined) ---
                feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
                # ---------------------------------------------
                
                # Apply scaled figsize and font size
                fig, ax = plt.subplots(figsize=plot_figsize)
                sns.barplot(x="Importance", y="Feature", data=feat_imp.head(15), ax=ax, color=accent_color_analysis, edgecolor="black") 
                ax.set_title("Top 15 Features", fontsize=plot_font_size, fontweight="bold")
                ax.set_xlabel("Importance", fontsize=plot_font_size * 0.6)
                ax.set_ylabel("Feature", fontsize=plot_font_size * 0.6)
                ax.tick_params(axis='x', labelsize=plot_font_size * 0.4)
                ax.tick_params(axis='y', labelsize=plot_font_size * 0.4)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ Could not generate feature importance: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # --- Delete test data now that analysis is complete ---
        del X_test, y_test 
        gc.collect() 
        # -----------------------------------------------------------


    # --- Inference Section (Analysis Section Color & Background) ---
    st.markdown(f"<div class='card-analysis'><div class='card-title-analysis'>🧪 Inference with Saved Model</div>", unsafe_allow_html=True)
    model_file = st.file_uploader("Upload trained model (.joblib)", type=["joblib"], key="inference_model")
    new_data_file = st.file_uploader("Upload new CSV/XLSX for prediction", type=["csv","xlsx"], key="inference_data")

    if model_file and new_data_file:
        try:
            loaded_model = joblib.load(model_file)
            if new_data_file.name.endswith(".csv"):
                new_df = pd.read_csv(new_data_file)
            else:
                new_df = pd.read_excel(new_data_file)
            
            # --- APPLY MEMORY OPTIMIZATION TO NEW DATA ---
            new_df = reduce_mem_usage(new_df)
            # ---------------------------------------------

            st.write("🔎 New Data Preview")
            try:
                st.dataframe(new_df.head(), use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display new data preview normally due to serialization issues. Falling back to plain text. Error: {e}")
                st.text(new_df.head().to_string())


            if target_col in new_df.columns:
                new_df = new_df.drop(columns=[target_col])
                
            # Drop datetime columns from new data as well
            datetime_cols_new = new_df.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns.tolist()
            if datetime_cols_new:
                new_df = new_df.drop(columns=datetime_cols_new)

            # Re-create the X data structure to match what the model expects
            if encoding_choice.startswith("One-hot encode"): # Check if one-hot was chosen during training
                # --- Use the stored feature_names list for alignment ---
                new_df = pd.get_dummies(new_df, drop_first=True)
                # Create missing columns and set to 0
                missing_cols = set(feature_names) - set(new_df.columns) 
                for col in missing_cols:
                    new_df[col] = 0
                
                # Align columns and drop extraneous ones based on the training features
                new_df = new_df[feature_names] 
                # ----------------------------------------------------------
            else:
                # If "Drop categorical" was chosen, no one-hot encoding is applied, but features must be aligned.
                new_df = new_df.drop(columns=[c for c in new_df.columns if c not in feature_names], errors='ignore')
                
            # Impute missing values in new data features using training data logic (mean)
            for col in new_df.columns:
                if new_df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(new_df[col]):
                        # Using mean of the new data for simplicity, though ideally should use training data mean.
                        new_df[col].fillna(new_df[col].mean(), inplace=True) 
            
            # Force GC after feature processing
            gc.collect()

            preds = loaded_model.predict(new_df)
            new_df["Prediction"] = preds

            st.write("📊 Predictions")
            try:
                st.dataframe(new_df.head(), use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display predictions DataFrame normally due to serialization issues. Falling back to plain text. Error: {e}")
                st.text(new_df.head().to_string())

            new_df.to_csv("predictions.csv", index=False)
            with open("predictions.csv", "rb") as f:
                st.download_button("⬇️ Download Predictions", f, file_name="predictions.csv")
        except Exception as e:
            st.error(f"⚠️ Inference failed: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)