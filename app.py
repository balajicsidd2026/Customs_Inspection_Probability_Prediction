import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import shap


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Customs Inspection Prediction System",
    layout="wide"
)

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------

model           = joblib.load("model/customs_inspection_cat_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
label_encoders  = joblib.load("model/label_encoders.pkl")

dataset = pd.read_csv(
    "dataset/customs_inspection_dataset_final_30000.csv"
)

test_data = pd.read_csv(
    "dataset/customs_inspection_test_dataset.csv"
)

test_data_1=test_data.drop(columns=["Inspection_Required"])
# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

.main {
    background-color: #f5f7fb;
}

h1 { color: #0b1b5e; font-weight: 800; }
h2, h3 { color: #0b1b5e; }

.stButton>button {
    background-color: #1565ff;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
    margin-top: 0px;
}

.result-box {
    background-color: #dbeafe;
    padding: 25px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    color: #0b4aa2;
}

.result-box-danger {
    background-color: #fee2e2;
    padding: 25px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    color: #b91c1c;
}

.stSelectbox div[data-baseweb="select"] > div { min-height: 40px; }
.stNumberInput input { height: 40px; }

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border-top: 5px solid #111827;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    text-align: left;
}
[data-testid="stMetricValue"] { font-size: 38px; font-weight: 700; color: #111827; }
[data-testid="stMetricLabel"] { font-size: 16px; color: gray; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown("""
<h1 style='font-size:40px; color:#1d2340; font-weight:600; margin-bottom:0px;'>
Customs Inspection Prediction System
</h1>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Prediction", "Analytics Dashboard","Inspection Result"])

# ===================================================
# TAB 1 — PREDICTION
# ===================================================

with tab1:

    col1, col2 = st.columns(2)

    # ---------- LEFT COLUMN ----------
    with col1:
        st.subheader("Shipment Information")
        shipment_id = st.selectbox(
            "Shipment ID",
            test_data_1["Shipment_ID"].unique(),help="Unique identifier for the shipment"
        )
        
        shipment_record = test_data_1[
            test_data_1["Shipment_ID"] == shipment_id
        ].iloc[0]
        
        shipment_date = shipment_record["Shipment_Date"]
        year = shipment_record["Year"]
        month = shipment_record["Month"]
        commodity_type = shipment_record["Commodity_Type"]
        commodity_risk_level = shipment_record["Commodity_Risk_Level"]
        cargo_season = shipment_record["Cargo_Season"]
        shipper_company = shipment_record["Shipper_Company_Name"]
        shipper_compliance_score = shipment_record["Shipper_Compliance_Score"]
        consignee_company = shipment_record["Consignee_Company_Name"]
        consignee_compliance_score = shipment_record["Consignee_Compliance_Score"]
        shipment_value_usd = shipment_record["Shipment_Value_USD"]
        weight_kg = shipment_record["Weight_KG"]
        previous_violations = shipment_record["Previous_Violations"]
        missing_documents = shipment_record["Missing_Documents"]
        
        shipment_date = pd.to_datetime(
            shipment_record["Shipment_Date"]
        )

        Year = shipment_date.year
        Month = shipment_date.month

        st.text_input(
            "Commodity Type",
            value=commodity_type,
            disabled=True,
            help="Category of goods being imported/exported"
        )
        
        shipment_value_usd = st.text_input(
            "Shipment Value (USD)",
            value=str(shipment_value_usd),
            disabled=True,help="Declared monetary value of the shipment"
        )

        weight_kg = st.text_input(
            "Weight (KG)",
            value=str(weight_kg),
            disabled=True,help="Total weight of the shipment in kilograms" 
        )

            
        st.text_input(
            "Commodity Risk Level",
            value=commodity_risk_level,
            disabled=True,help="Indicates the risk level associated with the commodity type"
        )
        
        
        # if Month in [3, 4]:
        #     cargo_season = "Ramadan"
        # elif Month in [6]:
        #     cargo_season = "Hajj"
        # elif Month in [11,12]:
        #     cargo_season = "Year_End_Peak"
        # else:
        #     cargo_season = "Regular"

            
        st.text_input(
            "Cargo Season",
            value=cargo_season,
            disabled=True,help="Indicates the seasonality of the cargo"
        )
        
    with col2:
        st.subheader("Company Information")

        shipper_company = st.text_input(
            "Shipper Company Name",
            value=shipper_company,
            disabled=True,help="Select the company sending the shipment"
        )

        consignee_company = st.text_input(
            "Consignee Company Name",
            value=consignee_company,
            disabled=True,help="Select the company receiving the shipment"
        )
        
        missing_documents = st.text_input(
            "Missing Documents",
            value=missing_documents,
            disabled=True,help="Indicates if any required documents are missing"
        )

        
        # shipper_compliance_score = shipper_score_map[shipper_company]
        # consignee_compliance_score = consignee_score_map[consignee_company]
        # previous_violations = violation_map[consignee_company]
        
        st.text_input(
            "Shipper Compliance Score",
            value=str(shipper_compliance_score),
            disabled=True,help="Compliance score of the shipper company"
        )
            
        st.text_input(
            "Consignee Compliance Score",
            value=str(consignee_compliance_score),
            disabled=True,help="Compliance score of the consignee company"
        )
        
        st.text_input(
            "Previous Violations",
            value=str(previous_violations),
            disabled=True,help="Number of previous violations by the consignee company"
        )

    # ---------- PREDICT BUTTON ----------
    if st.button("Predict Inspection Required"):

        # Build input dataframe
        input_data = pd.DataFrame({
            'year':                    [Year],
            'month':                   [Month],
            'Cargo_Season':             [cargo_season],
            
            'Commodity_Type':           [commodity_type],
            'Commodity_Risk_Level':     [commodity_risk_level],
        
            "Shipper_Company_Name": [shipper_company],
            "Shipper_Compliance_Score": [shipper_compliance_score],
            
            "Consignee_Company_Name": [consignee_company],
            "Consignee_Compliance_Score": [consignee_compliance_score],
            'Previous_Violations':      [previous_violations],    
            
            'Weight_KG':                [weight_kg],
            'shipment_value_usd':       [shipment_value_usd],     
            'Missing_Documents':        [missing_documents],

        })
        

        # Label encode categorical columns
        for col, le in label_encoders.items():
            if col in input_data.columns:
                try:
                    input_data[col] = le.transform(input_data[col])
                except ValueError:
                    input_data[col] = 0

        # Align columns to training order
        final_input = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Predict
        prediction  = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]

        result_col, table_col = st.columns([1, 1])

        with result_col:
            st.subheader("Prediction Results")

            if prediction == 1:
                st.markdown(f"""
                <div class="result-box-danger">
                ⚠️ Inspection Required<br><br>
                Risk Probability: {probability:.2%}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box">
                ✅ No Inspection Required<br><br>
                Risk Probability: {probability:.2%}
                </div>
                """, unsafe_allow_html=True)

        with table_col:
            st.subheader("Inspection Risk Factors")
            risk_factors = pd.DataFrame({
                "Feature": [
                    "Commodity Risk Level",
                    "Shipper Compliance Score",
                    "Consignee Compliance Score",
                    "Previous Violations",
                    "Missing Documents",
                    "Cargo Season",
                    "Shipment Value (USD)",
                    "Weight (KG)"
                ],
                "Value": [
                    commodity_risk_level,
                    shipper_compliance_score,
                    consignee_compliance_score,
                    previous_violations,
                    missing_documents,
                    cargo_season,
                    shipment_value_usd,
                    weight_kg
                ]
            })

            st.dataframe(
                risk_factors,
                use_container_width=True,
                hide_index=True
            )
# ===================================================
# TAB 2 — ANALYTICS DASHBOARD
# ===================================================

with tab2:

    # Prepare date column
    test_data['Shipment_Date'] = pd.to_datetime(
        test_data['Shipment_Date'], errors='coerce'
    )
    dataset_start_date = test_data['Shipment_Date'].min()
    dataset_end_date   = test_data['Shipment_Date'].max()

    # ---- FILTERS ----
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        from_date = st.date_input(
            "From Date",
            value=dataset_start_date,
            min_value=dataset_start_date,
            max_value=dataset_end_date
        )

    with filter_col2:
        to_date = st.date_input(
            "To Date",
            value=dataset_end_date,
            min_value=dataset_start_date,
            max_value=dataset_end_date
        )

    with filter_col3:
        quick_filter = st.selectbox(
            "Quick Filter",
            ["All Time", "Last 30 Days", "Last 90 Days",
            "Last 6 Months", "Last 1 Year"]
        )

    # ---- DATE FILTERING ----
    filtered_data = test_data.copy()
    today = filtered_data['Shipment_Date'].max()

    if quick_filter == "Last 30 Days":
        from_date = today - pd.Timedelta(days=30)
    elif quick_filter == "Last 90 Days":
        from_date = today - pd.Timedelta(days=90)
    elif quick_filter == "Last 6 Months":
        from_date = today - pd.DateOffset(months=6)
    elif quick_filter == "Last 1 Year":
        from_date = today - pd.DateOffset(years=1)

    filtered_data = filtered_data[
        (filtered_data['Shipment_Date'] >= pd.to_datetime(from_date))
        & (filtered_data['Shipment_Date'] <= pd.to_datetime(to_date))
    ]

    # ---- KPI VALUES ----
    total_shipments    = len(filtered_data)
    inspections_needed = filtered_data[filtered_data['Inspection_Required'] == 1].shape[0]
    inspection_rate    = (inspections_needed / total_shipments * 100) if total_shipments > 0 else 0
    missing_docs_count = filtered_data[filtered_data['Missing_Documents'] == 'Yes'].shape[0]
    avg_compliance     = filtered_data['Shipper_Compliance_Score'].mean()

    # ---- KPI CARDS ----
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric("Total Shipments",     f"{total_shipments:,}")
    with kpi2:
        st.metric("Inspection Rate",     f"{round(inspection_rate, 1)}%")
    with kpi3:
        st.metric("Inspections Needed",  f"{inspections_needed:,}")
    with kpi4:
        st.metric("Missing Documents",   f"{missing_docs_count:,}")
    with kpi5:
        st.metric("Avg Compliance Score", f"{round(avg_compliance, 1)}")

    # ---- ROW 1 ----
    chart1, chart2 = st.columns(2)

    # 1. Monthly Inspection Trend
    with chart1:
        monthly = (
            filtered_data
            .groupby(filtered_data["Shipment_Date"].dt.month_name().str[:3])
            ["Inspection_Required"]
            .mean()
            .reset_index()
        )
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["Shipment_Date"] = pd.Categorical(
            monthly["Shipment_Date"], categories=month_order, ordered=True
        )
        monthly = monthly.sort_values("Shipment_Date")
        monthly["Inspection_Required"] = (monthly["Inspection_Required"] * 100)

        fig1 = px.line(monthly, x="Shipment_Date", y="Inspection_Required", markers=True)
        fig1.update_traces(
            mode="lines+markers+text",
            text=monthly["Inspection_Required"].round(1),
            textposition="top center",
            line=dict(color="#0b66c3", width=3),
            marker=dict(size=8, color="#0b66c3"),
            textfont=dict(size=12, color="#111827")
        )
        fig1.update_layout(
            title="1. Monthly Inspection Trend",
            title_font_size=24, title_font_color="#111827",
            height=420, template="plotly_white",
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis=dict(title="Month", showgrid=False, tickfont=dict(size=14)),
            yaxis=dict(title="Inspection Rate (%)", gridcolor="#E5E7EB",
                       range=[0, monthly["Inspection_Required"].max() + 5],
                       tickfont=dict(size=14))
        )
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Commodity Type vs Inspection Count
    with chart2:
        comm_data = (
            filtered_data[filtered_data["Inspection_Required"] == 1]
            .groupby("Commodity_Type").size()
            .reset_index(name="Inspection_Count")
            .sort_values("Inspection_Count", ascending=False)
            .head(7)
        )
        colors = ["#1d4ed8","#2563eb","#3b82f6","#60a5fa","#93c5fd","#BFDBFE","#dbeafe"]
        comm_data["Color"] = colors[:len(comm_data)]

        fig2 = px.bar(comm_data, x="Commodity_Type", y="Inspection_Count", text="Inspection_Count")
        fig2.update_traces(marker_color=comm_data["Color"], textposition="outside")
        fig2.update_layout(
            title="2. Commodity Type vs Inspection Count",
            title_font_size=24, title_font_color="#111827",
            height=420, template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(title="Commodity Type", tickfont=dict(size=13)),
            yaxis=dict(title="Inspection Count", gridcolor="#E5E7EB",
                       tickfont=dict(size=10),
                       range=[0, comm_data["Inspection_Count"].max() + 100]),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ---- ROW 2 ----
    chart3, chart4, chart5 = st.columns(3)

    # 3. Missing Documents vs Inspection Rate
    with chart5:
        violation_data = (
            filtered_data
            .groupby("Commodity_Type")["Previous_Violations"]
            .mean()
            .reset_index()
            .sort_values("Previous_Violations", ascending=True)
        )

        violation_data["Previous_Violations"] = (
            violation_data["Previous_Violations"].round(2)
        )

        fig5 = px.bar(
            violation_data,
            x="Previous_Violations",
            y="Commodity_Type",
            orientation="h",
            color="Previous_Violations",
            text="Previous_Violations",
            color_continuous_scale="Blues"
        )

        fig5.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )

        fig5.update_layout(
            title="5. Commodity Type vs Average Violation Count",
            title_font_size=24,
            title_font_color="#111827",
            height=420,
            showlegend=False,
            coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Average Violation Count",
            yaxis_title="Commodity Type",
            xaxis=dict(
                gridcolor="#E5E7EB",
                tickfont=dict(size=14),
                fixedrange=True
            ),
            yaxis=dict(
                tickfont=dict(size=14),
                fixedrange=True
            )
        )

        st.plotly_chart(
            fig5,
            use_container_width=True,
            config={"displayModeBar": False}
        )
        
    # 4. Cargo Season Risk Distribution
    with chart4:
        season_data = (
            filtered_data
            .groupby("Cargo_Season")["Inspection_Required"]
            .mean().reset_index()
        )
        season_data["Risk_Pct"] = (season_data["Inspection_Required"] * 100).round(1)
        season_data = season_data.sort_values("Risk_Pct", ascending=False)

        fig4 = px.pie(season_data, names="Cargo_Season", values="Risk_Pct",
                      hole=0.55,
                      color_discrete_sequence=["#1d4ed8","#2563eb","#3b82f6","#60a5fa","#93c5fd"])
        fig4.update_traces(textposition='outside', textinfo='percent+label',
                           hoverinfo="skip", hovertemplate=None,
                           pull=[0.03, 0, 0, 0],
                           marker=dict(line=dict(color='white', width=2)))
        fig4.update_layout(
            title=dict(text="4. Cargo Season Risk Distribution", x=0,
                       font=dict(size=22, color="#111827")),
            height=420, template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=70, b=20), showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # 5. Commodity Risk Level vs Inspection Rate
    with chart3:
        risk_data = (
            filtered_data
            .groupby("Commodity_Risk_Level")["Inspection_Required"]
            .mean().reset_index()
        )
        risk_data["Rate"] = (risk_data["Inspection_Required"] * 100).round(1)
        risk_data["Commodity_Risk_Level"] = pd.Categorical(
            risk_data["Commodity_Risk_Level"],
            categories=["Low", "Medium", "High"], ordered=True
        )
        risk_data = risk_data.sort_values("Commodity_Risk_Level")

        fig3 = px.bar(risk_data, x="Rate", y="Commodity_Risk_Level",
                      orientation="h", text="Rate",
                      title="3. Commodity Risk Level vs Inspection Rate",
                      color="Commodity_Risk_Level",
                      color_discrete_map={"Low":"#bfdbfe","Medium":"#60a5fa","High":"#1d4ed8"})
        fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig3.update_layout(
            height=420, template="plotly_white",
            showlegend=False, paper_bgcolor="white", plot_bgcolor="white",
            title_font_size=20, title_font_color="#111827",
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis=dict(title="Inspection Rate (%)", gridcolor="#E5E7EB",
                       tickfont=dict(size=13), range=[0, 100]),
            yaxis=dict(title="", tickfont=dict(size=15))
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ===================================================
# TAB 3 — BULK INSPECTION ANALYSIS
# ===================================================

with tab3:

    st.subheader("Bulk Inspection Analysis")

    # Run Analysis Button
    if st.button("Run Inspection Analysis"):

        bulk_data = test_data_1.copy()

        # Encode categorical columns
        for col, encoder in label_encoders.items():

            if col in bulk_data.columns:

                try:
                    bulk_data[col] = encoder.transform(
                        bulk_data[col]
                    )

                except:
                    bulk_data[col] = 0

        # Arrange columns exactly as training
        bulk_data = bulk_data[feature_columns]

        # Predict
        predictions = model.predict(bulk_data)

        probabilities = model.predict_proba(
            bulk_data
        )[:, 1]

        # Create Results
        results = test_data_1.copy()

        results["Inspection_Required"] = predictions

        results["Inspection_Probability"] = (
            probabilities * 100
        ).round(2)

        # Save results in session state
        st.session_state["results"] = results

    # ==================================================
    # SHOW RESULTS AFTER BUTTON CLICK
    # ==================================================

    if "results" in st.session_state:

        results = st.session_state["results"]

        total_shipments = len(results)

        inspection_required = (
            results["Inspection_Required"] == 1
        ).sum()

        no_inspection = (
            results["Inspection_Required"] == 0
        ).sum()

        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric(
                "Total Shipments",
                f"{total_shipments:,}"
            )

        with k2:
            st.metric(
                "Inspection Required",
                f"{inspection_required:,}"
            )

        with k3:
            st.metric(
                "No Inspection Required",
                f"{no_inspection:,}"
            )

        st.success(
            f"Analysis Completed for {total_shipments:,} Shipments"
        )

        # ==================================================
        # FILTER INSPECTION REQUIRED SHIPMENTS
        # ==================================================

        inspection_shipments = results[
            results["Inspection_Required"] == 1
        ]

        inspection_shipments = inspection_shipments.sort_values(
            by="Inspection_Probability",
            ascending=False
        )

        st.subheader(
            "High Risk Shipments Requiring Inspection"
        )

        commodity_filter = st.selectbox(
            "Filter by Commodity Type",
            ["All"] +
            sorted(
                inspection_shipments[
                    "Commodity_Type"
                ].unique().tolist()
            )
        )

        if commodity_filter == "All":

            filtered_shipments = inspection_shipments

        else:

            filtered_shipments = inspection_shipments[
                inspection_shipments[
                    "Commodity_Type"
                ] == commodity_filter
            ]

        st.info(
            f"{len(filtered_shipments):,} shipments found"
        )

        st.dataframe(
            filtered_shipments[
                [
                    "Shipment_ID",
                    "Commodity_Type",
                    "Shipper_Company_Name",
                    "Consignee_Company_Name",
                    "Inspection_Probability"
                ]
            ],
            use_container_width=True
        )

        # ==================================================
        # DOWNLOAD REPORT
        # ==================================================

        csv = filtered_shipments.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Inspection Report",
            data=csv,
            file_name="inspection_required_shipments.csv",
            mime="text/csv"
        )