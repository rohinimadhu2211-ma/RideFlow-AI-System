import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="RideFlow AI System",

    layout="wide"
)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "rideflow_feature_engineered.csv"
)


# =========================================================
# LOAD MODELS
# =========================================================

demand_model = joblib.load(
    "ride_demand_prediction_model.pkl"
)

cancellation_model = joblib.load(
    "ride_cancellation_model.pkl"
)

sentiment_model = joblib.load(
    "sentiment_analysis_model.pkl"
)

tfidf = joblib.load(
    "tfidf_vectorizer.pkl"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "🚖 RideFlow AI System"
)

st.write(
    "AI-Based Intelligent Ride Management System"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(

    "Go To",

    [

        "Home",

        "Demand Prediction",

        "Cancellation Prediction",

        "Sentiment Analysis",

        "Ride Matching",

        "AI Chatbot"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.header(
        "🏠 Home"
    )

    st.write(

        """
Welcome to RideFlow AI Dashboard.

Modules Included:

✅ Ride Demand Prediction

✅ Ride Cancellation Prediction

✅ NLP Sentiment Analysis

✅ AI Ride Matching Assistant

✅ AI Chatbot
"""
    )


# =========================================================
# DEMAND PREDICTION
# =========================================================

elif page == "Demand Prediction":

    st.header(
        "📈 Ride Demand Prediction"
    )

    st.write(
        "Predict whether ride demand will be HIGH or LOW"
    )


    # ZONE NAMES

    zone_names = {

        "Anna Nagar": 0,

        "T Nagar": 1,

        "Adyar": 2,

        "Porur": 3,

        "OMR": 4,

        "Tambaram": 5,

        "Velachery": 6,

        "Nungambakkam": 7
    }


    # USER INPUTS

    pickup_zone_name = st.selectbox(

        "Pickup Zone",

        list(zone_names.keys())
    )

    hour = st.slider(
        "Hour",
        0,
        23,
        12
    )

    weekday = st.slider(
        "Weekday",
        0,
        6,
        3
    )


    # SAMPLE ROW

    sample = df.iloc[[0]].copy()


    # MODIFY INPUTS

    sample["pickup_zone"] = zone_names[
        pickup_zone_name
    ]

    sample["hour"] = hour

    sample["weekday"] = weekday


    # MATCH MODEL FEATURES

    required_features = demand_model.feature_names_in_

    sample = sample[required_features]


    # PREDICT BUTTON

    if st.button(
        "Predict Demand"
    ):

        prediction = demand_model.predict(
            sample
        )[0]


        st.subheader(
            "Prediction Result"
        )


        if prediction == 1:

            st.metric(

                "Demand Status",

                "HIGH"
            )

            st.error(

                "🔥 Peak-hour ride demand detected"
            )

        else:

            st.metric(

                "Demand Status",

                "LOW"
            )

            st.success(

                "✅ Normal ride demand"
            )


        # OPTIONAL CHART

        chart_data = pd.DataFrame({

            "Demand": [

                random.randint(20, 50),

                random.randint(40, 80),

                random.randint(60, 120),

                random.randint(30, 70)
            ]
        })


        st.line_chart(
            chart_data
        )


# =========================================================
# CANCELLATION PREDICTION
# =========================================================

elif page == "Cancellation Prediction":

    st.header(
        "❌ Ride Cancellation Prediction"
    )

    st.write(
        "Predict whether a ride may get cancelled"
    )


    # USER INPUTS

    eta = st.slider(
        "Estimated ETA (minutes)",
        1,
        60,
        15
    )

    fare = st.number_input(
        "Fare Price",
        50,
        1000,
        250
    )

    surge = st.slider(
        "Surge Multiplier",
        1.0,
        3.0,
        1.2
    )

    driver_rating = st.slider(
        "Driver Rating",
        1.0,
        5.0,
        4.0
    )


    # SAMPLE ROW

    sample = df.iloc[[0]].copy()


    # MODIFY INPUTS

    sample["estimated_eta_min"] = eta

    sample["fare_price"] = fare

    sample["surge_multiplier"] = surge

    sample["driver_rating"] = driver_rating


    # MATCH MODEL FEATURES

    required_features = cancellation_model.feature_names_in_

    sample = sample[required_features]


    # PREDICT BUTTON

    if st.button(
        "Predict Cancellation"
    ):


        prediction = cancellation_model.predict(
            sample
        )[0]


        st.subheader(
            "Prediction Result"
        )


        if prediction == 1:

            st.metric(

                "Cancellation Risk",

                "HIGH"
            )

            st.error(

                "❌ High probability of ride cancellation"
            )

        else:

            st.metric(

                "Cancellation Risk",

                "LOW"
            )

            st.success(

                "✅ Low probability of cancellation"
            )
  
        # CANCELLATION RISK GRAPH

        graph_data = pd.DataFrame({

            "Risk Score": [

                random.randint(10, 30),

                random.randint(20, 50),

                random.randint(40, 80),

                random.randint(60, 100)
            ]
        })


        st.subheader(
            "Cancellation Risk Trend"
        )

        st.line_chart(
            graph_data
        )

# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

elif page == "Sentiment Analysis":

    st.header(
        "💬 NLP Sentiment Analysis"
    )

    st.write(
        "Analyze customer feedback sentiment using AI"
    )


    # USER INPUT

    review = st.text_area(

        "Enter Customer Review"
    )


    # ANALYZE BUTTON

    if st.button(

        "Analyze Sentiment"
    ):


        # TFIDF TRANSFORM

        review_tfidf = tfidf.transform(

            [review]
        )


        # PREDICTION

        prediction = sentiment_model.predict(

            review_tfidf
        )[0]


        # RESULT TITLE

        st.subheader(
            "Prediction Result"
        )


        # POSITIVE

        if prediction == "Positive":

            st.success(
                "😊 Positive Sentiment Detected"
            )

            st.balloons()

            st.metric(

                "Sentiment",

                "POSITIVE"
            )

            st.write(

                """
Customer had a good ride experience.

✔ Driver behavior positive

✔ Ride quality satisfactory

✔ Customer likely to rate highly
"""
            )


        # NEGATIVE

        elif prediction == "Negative":

            st.error(
                "😡 Negative Sentiment Detected"
            )

            st.metric(

                "Sentiment",

                "NEGATIVE"
            )

            st.write(

                """
Customer faced ride issues.

❌ Poor ride experience

❌ Possible complaint risk

❌ Driver/service improvement needed
"""
            )


        # NEUTRAL

        else:

            st.warning(
                "😐 Neutral Sentiment Detected"
            )

            st.metric(

                "Sentiment",

                "NEUTRAL"
            )

            st.write(

                """
Customer feedback is neutral.

ℹ No strong positive or negative opinion

ℹ Standard ride experience
"""
            )


        # =================================================
        # SENTIMENT GRAPH
        # =================================================

        sentiment_data = pd.DataFrame({

            "Sentiment Score": [

                random.randint(20, 40),

                random.randint(40, 60),

                random.randint(60, 90)
            ]
        })


        st.subheader(
            "Sentiment Trend"
        )

        st.bar_chart(
            sentiment_data
        )


        # =================================================
        # SAMPLE TEST REVIEWS
        # =================================================

        st.subheader(
            "Example Test Reviews"
        )

        st.info(

            """
😊 Positive:
Driver was very friendly and professional

😡 Negative:
Very bad ride experience and rude driver

😐 Neutral:
Ride completed successfully
"""
        ) 
# =========================================================
# RIDE MATCHING ASSISTANT
# =========================================================

elif page == "Ride Matching":

    st.header(
        "🚕 AI Ride Matching Assistant"
    )

    st.write(
        "AI-based smart driver recommendation system"
    )


    # =====================================================
    # USER INPUTS
    # =====================================================

    required_eta = st.slider(
        "Preferred ETA",
        1,
        20,
        5
    )

    minimum_rating = st.slider(
        "Minimum Driver Rating",
        1.0,
        5.0,
        4.0
    )


    # =====================================================
    # CREATE DRIVER DATA
    # =====================================================

    driver_data = pd.DataFrame({

        "driver_id": np.arange(
            1000,
            1010
        ),

        "driver_rating": np.random.uniform(
            3.5,
            5.0,
            10
        ),

        "eta": np.random.randint(
            2,
            20,
            10
        ),

        "cancellation_risk": np.random.uniform(
            0,
            1,
            10
        )
    })


    # =====================================================
    # FILTER DRIVERS
    # =====================================================

    filtered_drivers = driver_data[

        driver_data["driver_rating"]
        >= minimum_rating
    ]


    # =====================================================
    # MATCHING SCORE
    # =====================================================

    filtered_drivers["matching_score"] = (

        (filtered_drivers["driver_rating"] * 0.5)

        +

        (
            (1 / (filtered_drivers["eta"] + 1)) * 10
        )

        +

        (
            (1 - filtered_drivers["cancellation_risk"]) * 5
        )
    )


    # =====================================================
    # TOP DRIVER RECOMMENDATIONS
    # =====================================================

    top_drivers = filtered_drivers.sort_values(

        by="matching_score",

        ascending=False
    ).head(5)


    # =====================================================
    # SHOW RESULTS
    # =====================================================

    st.subheader(
        "Top Driver Recommendations"
    )

    st.success(
        "Best drivers selected using AI matching score"
    )


    st.dataframe(

        top_drivers[

            [

                "driver_id",

                "driver_rating",

                "eta",

                "cancellation_risk",

                "matching_score"
            ]
        ]
    )


    # =====================================================
    # BEST DRIVER
    # =====================================================

    best_driver = top_drivers.iloc[0]


    st.subheader(
        "Best Recommended Driver"
    )

    st.metric(

        "Driver ID",

        int(best_driver["driver_id"])
    )

    st.write(

        f"""
⭐ Driver Rating:
{round(best_driver['driver_rating'], 2)}

⏱ ETA:
{int(best_driver['eta'])} minutes

⚠ Cancellation Risk:
{round(best_driver['cancellation_risk'], 2)}

🏆 Matching Score:
{round(best_driver['matching_score'], 2)}
"""
    )


    # =====================================================
    # MATCHING SCORE GRAPH
    # =====================================================

    st.subheader(
        "Driver Matching Scores"
    )

    st.bar_chart(

        top_drivers.set_index(

            "driver_id"
        )["matching_score"]
    )


    # =====================================================
    # INSIGHTS
    # =====================================================

    st.info(

        """
AI Matching Logic Uses:

✅ Driver Rating

✅ ETA

✅ Cancellation Risk

✅ Smart Ranking Algorithm
"""
    )

# =========================================================
# AI CHATBOT WITH MULTILINGUAL SUPPORT
# =========================================================

elif page == "AI Chatbot":

    st.header(
        "🤖 RideFlow AI Assistant"
    )

    st.write(
        "AI-powered multilingual ride assistant"
    )


    # =====================================================
    # LANGUAGE SELECTION
    # =====================================================

    language = st.selectbox(

        "🌐 Select Language",

        [

            "English",

            "Hindi",

            "Tamil"
        ]
    )


    # =====================================================
    # USER INPUT
    # =====================================================

    query = st.text_input(
        "Ask Your Question"
    )


    # =====================================================
    # PROCESS QUERY
    # =====================================================

    if query:


        with st.spinner(
            "AI Assistant Thinking..."
        ):


            text = query.lower()


            # =================================================
            # ENGLISH SUPPORT
            # =================================================

            if language == "English":


                if "driver" in text and "rating" in text:

                    response = (
                        "⭐ Your assigned driver has a 4.8 rating."
                    )


                elif "driver details" in text:

                    response = (
                        "🧍 Driver Name: Rahul | Rating: 4.8 | Vehicle: KA01AB1234"
                    )


                elif (

                    "driver" in text

                    or "location" in text
                ):

                    response = (
                        "🚖 Your driver is near MG Road and will arrive in 5 minutes."
                    )


                elif "eta" in text:

                    response = (
                        "⏱ Estimated arrival time is 5 minutes."
                    )


                elif "ride status" in text:

                    response = (
                        "🚖 Your ride is currently in progress."
                    )


                elif "book ride" in text:

                    response = (
                        "✅ Ride booking request submitted successfully."
                    )


                elif "pickup" in text:

                    response = (
                        "📍 Pickup location confirmed successfully."
                    )


                elif "drop" in text:

                    response = (
                        "🗺 Destination location updated successfully."
                    )


                elif "track" in text:

                    response = (
                        "📍 Live ride tracking is currently active."
                    )


                elif "navigation" in text:

                    response = (
                        "🛣 Fastest route navigation activated."
                    )


                elif (

                    "price" in text

                    or "fare" in text

                    or "surge" in text
                ):

                    response = (
                        "💰 Surge pricing is active due to high demand."
                    )


                elif (

                    "payment" in text

                    or "pay" in text
                ):

                    response = (
                        "💳 Payment completed successfully."
                    )


                elif "invoice" in text:

                    response = (
                        "🧾 Ride invoice has been sent to your email."
                    )


                elif "refund" in text:

                    response = (
                        "💰 Your refund request has been initiated."
                    )


                elif (

                    "cancel" in text

                    or "cancellation" in text
                ):

                    response = (
                        "❌ Your ride cancellation request has been processed."
                    )


                elif "traffic" in text:

                    response = (
                        "🚦 Heavy traffic detected near pickup location."
                    )


                elif "weather" in text:

                    response = (
                        "🌧 Rain detected. Ride demand may increase."
                    )


                elif "available drivers" in text:

                    response = (
                        "🚘 12 drivers are currently available nearby."
                    )


                elif "earnings" in text:

                    response = (
                        "💵 Driver earnings for today are ₹2,450."
                    )


                elif "coupon" in text:

                    response = (
                        "🎁 Coupon RIDE20 applied successfully."
                    )


                elif "network" in text:

                    response = (
                        "📶 Weak network detected. Trying to reconnect."
                    )


                elif "battery" in text:

                    response = (
                        "🔋 Driver device battery status is stable."
                    )


                elif "lost item" in text:

                    response = (
                        "👜 Lost item report has been submitted."
                    )


                elif "accident" in text:

                    response = (
                        "🚨 Emergency accident assistance activated."
                    )


                elif "safety" in text:

                    response = (
                        "🛡 Ride safety monitoring is active."
                    )


                elif "support" in text:

                    response = (
                        "📞 Customer support has been notified."
                    )


                elif "emergency" in text:

                    response = (
                        "🚨 Emergency support team has been alerted."
                    )


                elif (

                    "hello" in text

                    or "hi" in text
                ):

                    response = (
                        "👋 Hello! Welcome to RideFlow AI."
                    )


                elif "thank" in text:

                    response = (
                        "😊 You're welcome!"
                    )


                elif "bye" in text:

                    response = (
                        "👋 Thank you for using RideFlow AI."
                    )


                else:

                    response = (
                        "🤖 Sorry, I could not understand your request."
                    )


            # =================================================
            # HINDI SUPPORT
            # =================================================

            elif language == "Hindi":


                if "driver" in text:

                    response = (
                        "🚖 आपका ड्राइवर 5 मिनट में पहुंचेगा।"
                    )


                elif "payment" in text:

                    response = (
                        "💳 आपका पेमेंट सफल हुआ।"
                    )


                elif "cancel" in text:

                    response = (
                        "❌ आपकी राइड कैंसल कर दी गई है।"
                    )


                elif "refund" in text:

                    response = (
                        "💰 आपका रिफंड शुरू कर दिया गया है।"
                    )


                elif "support" in text:

                    response = (
                        "📞 ग्राहक सहायता टीम को सूचित किया गया है।"
                    )


                elif "traffic" in text:

                    response = (
                        "🚦 आपके क्षेत्र में भारी ट्रैफिक है।"
                    )


                elif "weather" in text:

                    response = (
                        "🌧 बारिश के कारण मांग बढ़ सकती है।"
                    )


                elif "rating" in text:

                    response = (
                        "⭐ ड्राइवर की रेटिंग 4.8 है।"
                    )


                elif "track" in text:

                    response = (
                        "📍 लाइव राइड ट्रैकिंग चालू है।"
                    )


                elif "ride status" in text:

                    response = (
                        "🚖 आपकी राइड जारी है।"
                    )


                elif "book ride" in text:

                    response = (
                        "✅ आपकी राइड सफलतापूर्वक बुक हो गई है।"
                    )


                elif "coupon" in text:

                    response = (
                        "🎁 कूपन सफलतापूर्वक लागू किया गया।"
                    )


                elif "earnings" in text:

                    response = (
                        "💵 आज की ड्राइवर कमाई ₹2450 है।"
                    )


                elif "hello" in text:

                    response = (
                        "👋 नमस्ते! RideFlow AI में आपका स्वागत है।"
                    )


                else:

                    response = (
                        "🤖 क्षमा करें, मैं आपकी बात समझ नहीं पाया।"
                    )


            # =================================================
            # TAMIL SUPPORT
            # =================================================

            elif language == "Tamil":


                if "driver" in text:

                    response = (
                        "🚖 உங்கள் டிரைவர் 5 நிமிடங்களில் வருவார்."
                    )


                elif "payment" in text:

                    response = (
                        "💳 உங்கள் கட்டணம் வெற்றிகரமாக முடிந்தது."
                    )


                elif "cancel" in text:

                    response = (
                        "❌ உங்கள் பயணம் ரத்து செய்யப்பட்டது."
                    )


                elif "refund" in text:

                    response = (
                        "💰 உங்கள் பணம் திரும்ப வழங்கப்படும்."
                    )


                elif "support" in text:

                    response = (
                        "📞 வாடிக்கையாளர் ஆதரவு தொடர்பு கொள்ளப்பட்டது."
                    )


                elif "traffic" in text:

                    response = (
                        "🚦 உங்கள் பகுதியில் அதிக போக்குவரத்து உள்ளது."
                    )


                elif "weather" in text:

                    response = (
                        "🌧 மழையால் ரைடு தேவைகள் அதிகரிக்கலாம்."
                    )


                elif "rating" in text:

                    response = (
                        "⭐ டிரைவர் மதிப்பீடு 4.8 ஆக உள்ளது."
                    )


                elif "track" in text:

                    response = (
                        "📍 நேரடி ரைடு கண்காணிப்பு செயல்பாட்டில் உள்ளது."
                    )


                elif "ride status" in text:

                    response = (
                        "🚖 உங்கள் பயணம் நடைபெற்று வருகிறது."
                    )


                elif "book ride" in text:

                    response = (
                        "✅ உங்கள் ரைடு வெற்றிகரமாக பதிவு செய்யப்பட்டது."
                    )


                elif "coupon" in text:

                    response = (
                        "🎁 கூப்பன் வெற்றிகரமாக பயன்படுத்தப்பட்டது."
                    )


                elif "earnings" in text:

                    response = (
                        "💵 இன்றைய டிரைவர் வருமானம் ₹2450."
                    )


                elif "hello" in text:

                    response = (
                        "👋 வணக்கம்! RideFlow AI-க்கு வரவேற்கிறோம்."
                    )


                else:

                    response = (
                        "🤖 மன்னிக்கவும், உங்கள் கேள்வியை புரிந்து கொள்ள முடியவில்லை."
                    )


        # =====================================================
        # SHOW RESPONSE
        # =====================================================

        st.markdown(

            f"""
<div style="
background-color:#F1F0F0;
padding:15px;
border-radius:12px;
margin-top:20px;
color:black;
font-size:17px;
">

🤖 <b>AI Assistant:</b><br><br>

{response}

</div>
""",

            unsafe_allow_html=True
        )


 