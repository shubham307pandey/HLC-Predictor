import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments, rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from rasar import calculate_descriptor
import joblib
from rdkit import DataStructs
from rdkit.Chem import AllChem



#inputs
files=joblib.load("Final_Models.joblib")
qsar_model = files[0]
rasar_model = files[1]
qsar_descriptors = files[2]
rasar_descriptors = files[3]
qsar_tr = files[4]
rasar_tr = files[5]


def standerdization(df1, df2):
    df1 = df1.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    df2 = df2.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    avg = df1.mean()
    stdev = df1.std().replace(0, 1).fillna(1)

    std_df1 = (df1 - avg) / stdev
    std_df2 = (df2 - avg) / stdev

    return (
        std_df1.replace([np.inf, -np.inf], np.nan).fillna(0),
        std_df2.replace([np.inf, -np.inf], np.nan).fillna(0)
    )

def std_ad(df: pd.DataFrame, avg, stdev):
    """
    Standard AD calculation using mean and std of training data.
    """

    df = df.copy()
    #mean = pd.Series(avg)
    #std = pd.Series(stdev)
    mean = avg
    std = stdev

    std_df = np.abs(df - mean) / std
    print(std_df)

    sk_score = std_df.mean(axis=1) + (1.28 * std_df.std(axis=1))
    max_std = std_df.max(axis=1)
    min_std = std_df.min(axis=1)

    ad_status = np.full(len(df), "Inside AD", dtype=object)

    # Conditions
    cond1 = (max_std > 3) & (min_std > 3)
    cond2 = ((max_std > 3) & (min_std < 3)) & (sk_score > 3)
    cond3 = ((max_std < 3) & (min_std > 3)) & (sk_score > 3)

    ad_status[cond1 | cond2 | cond3] = "Outside AD"

    df["AD Status"] = ad_status

    return df

def calc_ecfp(
    smiles_list,
    radius=2,
    nBits=2048,
    fp_type="bits",     # "bits" or "count"
    full_output=True
):
    fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            fps.append(np.zeros(nBits, dtype=int) if full_output else {})
            continue

        # ---------- BIT-BASED ECFP ----------
        if fp_type == "bits":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=radius,
                nBits=nBits
            )

            if full_output:
                arr = np.zeros(nBits, dtype=int)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            else:
                fps.append({i: 1 for i in fp.GetOnBits()})

        # ---------- COUNT-BASED ECFP ----------
        elif fp_type == "count":
            fp = AllChem.GetMorganFingerprint(
                mol,
                radius=radius,
                useCounts=True
            )

            if full_output:
                arr = np.zeros(nBits, dtype=int)
                for k, v in fp.GetNonzeroElements().items():
                    arr[k % nBits] += v   # hash folding
                fps.append(arr)
            else:
                # folded keys
                folded = {}
                for k, v in fp.GetNonzeroElements().items():
                    folded[k % nBits] = folded.get(k % nBits, 0) + v
                fps.append(folded)

        else:
            raise ValueError("fp_type must be 'bits' or 'count'")

    # ---------- CREATE DATAFRAME ----------
    if full_output:
        df = pd.DataFrame(
            fps
        )
        if fp_type == "bits":
            df.columns = [f"ECFP4_{c+1}" for c in df.columns]
        else:
            df.columns = [f"ECFP4(Count)_{c+1}" for c in df.columns]
    else:
        df = pd.DataFrame(fps).fillna(0).astype(int)
        if fp_type == "bits":
            df.columns = [f"ECFP4_{c+1}" for c in df.columns]
        else:
            df.columns = [f"ECFP4(Count)_{c+1}" for c in df.columns]

    return df

def _safe_numeric_dataframe(df, required_columns=None):
    """
    Convert descriptor values to numeric values.
    Text, None, NaN, inf, and non-convertible values are replaced by 0.
    Missing required descriptor columns are also created with value 0.
    """
    df = df.copy()

    if required_columns is not None:
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        df = df.loc[:, list(required_columns)]

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            df[col] = 0

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def rd_des(mol_list):
    """
    Calculate all RDKit descriptors safely.
    Any descriptor calculation error or non-numeric result is replaced by 0.
    """
    descriptor_names = [name for name, func in Descriptors.descList]
    cal_des = []

    for mol in mol_list:
        values = {name: 0 for name in descriptor_names}

        if mol is not None:
            try:
                calculated = Descriptors.CalcMolDescriptors(
                    mol, missingVal=0, silent=True
                )

                if calculated is not None:
                    for name in descriptor_names:
                        value = calculated.get(name, 0)
                        try:
                            value = pd.to_numeric(value, errors="coerce")
                            if pd.isna(value) or not np.isfinite(float(value)):
                                value = 0
                        except Exception:
                            value = 0
                        values[name] = value
            except Exception:
                pass

        cal_des.append(values)

    return _safe_numeric_dataframe(
        pd.DataFrame(cal_des, columns=descriptor_names)
    )


def final_descriptors(smiles_list):
    """
    Calculate RDKit descriptors, ECFP4 binary fingerprints and ECFP4 count
    fingerprints. Any calculation error is replaced with zero.
    """
    mols = []

    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        except Exception:
            mol = None
        mols.append(mol)

    try:
        rdkit_desc = rd_des(mols)
    except Exception:
        rdkit_desc = pd.DataFrame(
            np.zeros((len(smiles_list), len(Descriptors.descList))),
            columns=[name for name, func in Descriptors.descList]
        )

    try:
        ecfp_desc_bit = calc_ecfp(
            smiles_list=smiles_list,
            radius=2,
            fp_type="bits",
            full_output=True
        )
    except Exception:
        ecfp_desc_bit = pd.DataFrame(
            np.zeros((len(smiles_list), 2048), dtype=int),
            columns=[f"ECFP4_{i+1}" for i in range(2048)]
        )

    try:
        ecfp_desc_count = calc_ecfp(
            smiles_list=smiles_list,
            radius=2,
            fp_type="count",
            full_output=True
        )
    except Exception:
        ecfp_desc_count = pd.DataFrame(
            np.zeros((len(smiles_list), 2048), dtype=int),
            columns=[f"ECFP4(Count)_{i+1}" for i in range(2048)]
        )

    return _safe_numeric_dataframe(
        pd.concat(
            [
                _safe_numeric_dataframe(rdkit_desc),
                _safe_numeric_dataframe(ecfp_desc_bit),
                _safe_numeric_dataframe(ecfp_desc_count)
            ],
            axis=1
        )
    )



#datasets
qsar_xtr = qsar_tr.iloc[:, :-1]
qsar_ytr = qsar_tr.iloc[:, -1]
rasar_xtr = rasar_tr.iloc[:, :-1]
rasar_ytr = rasar_tr.iloc[:, -1]




##Custom Font
st.set_page_config(
    page_title="HLC-Predictor: A Henry's Law Constant Prediction Tool",
    layout="wide",
    page_icon="🧪"
)

# Initialize session-state variables before any buttons use them.
if "show_info" not in st.session_state:
    st.session_state.show_info = False

if "show_manual" not in st.session_state:
    st.session_state.show_manual = False

custom_css = """
<style>

    .stApp {
        font-family: 'Times New Roman', serif;
    }

    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        font-family: 'Times New Roman', serif !important;
        font-style: italic !important;
    }

    
    .stButton > button,
    .stDownloadButton > button {
        font-family: 'Times New Roman', serif !important;
        font-style: italic !important;
        font-weight: bold;
    }

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


st.sidebar.image("Henry Constant Image.png", width=400)
Model_selection= st.sidebar.selectbox("Select Regression Model", ["RR-QSPR Model", "LSVM q-RASPR Model"])

#Model Training
def prediction(data, type):
    if type == "RR-QSPR Model":
        try:
            pred=qsar_model.predict(data)

        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None


    elif type == "LSVM q-RASPR Model":
        try:
            pred=rasar_model.predict(data)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None
    else:
        st.error(f"Invalid model selection: {type}")
        return None

    return pred

def main():
    st.title("HLC-Predictor: A Henry's Law Constant Prediction Tool")
    st.markdown("""
<div style="
    font-family: 'Times New Roman', serif;
    font-size: 20px;
    font-style: italic;
    line-height: 1.6;
">*An open web-service to Predict Henry's Law Constant for volatality using structural, physicochemical, fingerprints, and similarity-based features.</div>""", unsafe_allow_html=True)
    
    st.markdown("""
<div style="
    font-family: 'Times New Roman', serif;
    font-size: 20px;
    font-style: italic;
">
*The HLC values are <b>Log atm·m³/mol</b>.
</div>
""", unsafe_allow_html=True)
   

    # Input method
    input_method = st.radio("Input Method", ["Single SMILES", "Upload Excel"])

    
    if input_method == "Single SMILES":
        smiles = st.text_input("Enter SMILES string:")
        smiles_list = [smiles]

    
    else:
        uploaded_file = st.file_uploader("Upload Excel with 'SMILES' column", type=['xlsx'])
        if uploaded_file:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                pass
            if 'SMILES' in df.columns:
                smiles_list = df['SMILES'].dropna().tolist()
                st.success(f"Loaded {len(smiles_list)} compounds")
            else:
                st.error("File must have a 'SMILES' column")

    
    if st.button("Predict HLC for input chemicals"):
        calc_desc = final_descriptors(smiles_list)

        # Force the descriptors required by the QSPR model to be numeric.
        # Missing columns, text, None, NaN and infinite values become zero.
        qsar_desc = _safe_numeric_dataframe(
            calc_desc,
            required_columns=qsar_descriptors
        )

        #qsar_desc.to_excel("qsarfile.xlsx", index=False)

        if Model_selection == "RR-QSPR Model":
            __, qsar_desc_std = standerdization(
                df1=qsar_tr.iloc[:, :-1],
                df2=qsar_desc
            )

            qsar_desc_std = _safe_numeric_dataframe(
                qsar_desc_std,
                required_columns=qsar_descriptors
            )

            pred = prediction(qsar_desc_std, Model_selection)
            pred1 = (pred * qsar_ytr.std()) + qsar_ytr.mean()

            ad_status = std_ad(
                qsar_desc,
                avg=qsar_tr.iloc[:, :-1].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0).mean(),
                stdev=qsar_tr.iloc[:, :-1].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0).std().replace(0, 1).fillna(1)
            )

        if Model_selection == "LSVM q-RASPR Model":
            try:
                __, rasar_ds = calculate_descriptor(
                    df1=qsar_tr,
                    df2=qsar_desc,
                    method="Laplacian Kernel",
                    ctc=10,
                    gamma=0.25
                )
            except Exception as e:
                st.warning(
                    "q-RASPR descriptor calculation encountered an error. "
                    "Error/unavailable descriptors will be replaced with 0. "
                    f"Details: {e}"
                )
                rasar_ds = pd.DataFrame(index=qsar_desc.index)

            # Required q-RASPR similarity/error descriptors.
            required_rasar_features = [
                "RA_function",
                "MaxNeg",
                "MaxPos",
                "sm2"
            ]

            for feature in required_rasar_features:
                if feature not in rasar_ds.columns:
                    rasar_ds[feature] = 0

            test_rasar = rasar_ds[required_rasar_features].copy()
            test_rasar.columns = [
                "RA function(LK)",
                "MaxNeg(LK)",
                "MaxPos(LK)",
                "sm2(LK)[Banerjee-Roy similarity coefficient 2]"
            ]

            test_rasar = _safe_numeric_dataframe(test_rasar)
            #test_rasar.to_excel("rasarfile.xlsx", index=False)

            test_rasar_df = pd.concat(
                [qsar_desc, test_rasar],
                axis=1
            )

            # Exact descriptor set/order required by the trained q-RASPR model.
            rasar_desc = _safe_numeric_dataframe(
                test_rasar_df,
                required_columns=rasar_descriptors
            )

            __, rasar_desc_std = standerdization(
                df1=rasar_tr.iloc[:, :-1],
                df2=rasar_desc
            )

            rasar_desc_std = _safe_numeric_dataframe(
                rasar_desc_std,
                required_columns=rasar_descriptors
            )

            pred = prediction(rasar_desc_std, Model_selection)
            pred1 = (pred * rasar_ytr.std()) + rasar_ytr.mean()

            ad_status = std_ad(
                test_rasar_df,
                avg=rasar_tr.iloc[:, :-1].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0).mean(),
                stdev=rasar_tr.iloc[:, :-1].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0).std().replace(0, 1).fillna(1)
            )

        result_df = pd.DataFrame(pred1, columns=["Predicted HLC"], index=smiles_list)
        result_df["AD Status"] = ad_status["AD Status"].values
        result_df.index.name = "SMILES"
        st.success("Predictions Complete!")
        st.dataframe(result_df)

        # Download
        csv = result_df.to_csv(index=True)
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

    st.sidebar.header("About")
    st.sidebar.info("""
    - **RR QSPR Model**: Uses 26 structural, physicochemical, and fingerprint descriptors
    - **LSVM q-RASPR Model**: Uses 16 structural, physicochemical, fingerprint, similarity, and error-based descriptors
    - Structural, Physicochemical, & Fingerprint descriptor are calculated using RDKit
    """)

    st.sidebar.markdown("### Requirements")
    st.sidebar.code("""
pip install streamlit rdkit scikit-learn pandas openpyxl rasar
    """)

if __name__ == "__main__":
    main()



st.divider()
col1, col2, col3 = st.columns([1.5, 6, 1.5])

with col1:
    if st.button("ℹ️ Contact Info", use_container_width=True):
        st.session_state.show_info = not st.session_state.show_info
        st.session_state.show_manual = False

with col3:
    if st.button("📖 User Manual", use_container_width=True):
        st.session_state.show_manual = not st.session_state.show_manual
        st.session_state.show_info = False


if st.session_state.show_info:
    st.markdown("""
    <div class="panel-box contact">
 
      <p class="panel-title contact"> Contact Information</p>
 
      <p class="sec-heading contact"> Contact </p>
      <div class="contact-card" style='font-family: "Times New Roman", serif; font-size: 18px; font-style: italic;'>
        <strong>Prof. (Dr.) Kunal Roy</strong><br>
        <a href="mailto:kunal.roy@jadavpuruniversity.in">kunal.roy@jadavpuruniversity.in</a><br>
        Drug Theoretics and Cheminformatics Laboratory<br>
        Jadavpur University, Kolkata, IN
      </div>
      </p>
    """, unsafe_allow_html=True)

if st.session_state.show_manual:
      with open("Manual.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

        st.markdown("""
    <div style='padding: 10px; background-color: #e6f3ff;
                border-radius: 8px; margin-top: 10px;'>
        <h4>User Manual</h4>
        <p>Click below to download the user manual.</p>
    </div>
    """, unsafe_allow_html=True)

        st.download_button(
        label="📖 Download User Manual",
        data=PDFbyte,
        file_name="User_Manual.pdf",
        mime="application/pdf"
    )
