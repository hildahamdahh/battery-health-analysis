🔋 **Battery Health Analysis (LFP + NMC)**

**Optimizing Charging Cycles Through Data Analysis**

This project explores how charging cycles and depth of discharge affect battery health (State of Health / SoH) using two battery types: LFP and NMC.  
It combines data analysis, visualization, and a simple predictive RNN model to generate actionable insights for optimizing battery lifespan.

📊 **Project Overview**  
Objective:  
Analyze large-scale battery data to identify how charging habits and usage patterns impact battery degradation, and provide recommendations to extend battery life.  
Dataset Size: ~1.5 million rows → cleaned to ~420k usable rows.  
Battery Types:  
- LFP (Lithium Iron Phosphate)
- NMC (Nickel Manganese Cobalt)  


🧹 **Data Cleaning Example**  
Filtering unrealistic voltage and current values:  
df_clean = df[(df['Voltage'] >= 2500) & (df['Voltage'] <= 4200)]  
df_clean = df_clean[(df['Current'] >= 0.1) & (df['Current'] <= 1000)]  
print("Cleaned dataset:", df_clean.shape)

🔍 **Exploratory Data Analysis**  
- Trend: SoH consistently decreases with higher cycle count → natural degradation.  
- Correlation:  
Capacity ↔ SoH = +1.0  
DoD ↔ SoH = -1.0  
Voltage ↔ SoH = -0.96  
Distribution: Most batteries maintain >95% SoH, but deep discharges accelerate drop.  
- Key Findings:  
Optimal charging range: 20–80% SoC  
Ideal daily cycles: 1–2 cycles/day  
Deep discharge (>80% DoD) significantly accelerates degradation.

🧠 **RNN Modeling**  
A simple Recurrent Neural Network (RNN) was trained to predict SoH trends based on historical sensor data.  
This model validates the same behavioral patterns observed in EDA.  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import SimpleRNN, Dense  

model = Sequential([  
      SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'),  
      Dense(1, activation='linear')])  
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=30, batch_size=32)  

Results:  
RNN achieved lowest error (MAE ≈ 0.7%, MAPE ≈ 2.3%).  
Confirms strong influence of Depth of Discharge and Capacity on SoH degradation.

💡 **Recommendations**  
For Users:  
Charge between 20–80% for longer battery life.  
Limit to 1–2 charge cycles per day.  
Avoid full discharge (<10% SoC).  
For Businesses / Engineers:  
Implement dashboards (Power BI/Tableau) to monitor SoH in real time.  
Focus on tracking DoD and Capacity as key predictive features.  
Add temperature data for more robust modeling in future.

🧰 **Tools & Libraries**  

Data Wrangling -> Python (Pandas, NumPy), Excel  
Visualization -> Matplotlib, Seaborn, Power BI  
Modeling -> TensorFlow/Keras, scikit-learn  
Environment -> Jupyter Notebook  
Version Control -> Git, GitHub

📂 **Project Files**  
File	Description   
Model_SoH_LFP_RNN.ipynb	Analysis and RNN model for LFP battery data  
Model_SoH_NMC_RNN.ipynb	Analysis and model comparison for NMC battery data  
requirements.txt	List of required Python libraries  


🧾 **Key Takeaways**  
Handled large-scale sensor data (1.5M+ rows) efficiently.  
Performed cleaning, feature engineering, and visualization to find degradation patterns.  
Delivered actionable recommendations for end-users and engineers.  
Added RNN model to validate analytical insights (advanced but lightweight). 

👩‍💻 **Author**  
Hilda Hamdah H  
📍 Data Analyst  
🔗 https://www.linkedin.com/in/hilda-hamdah-h/  
📧 hildahamdahusniyyah22@gmail.com
