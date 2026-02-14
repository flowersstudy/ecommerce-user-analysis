from rfm_analysis import run_rfm_analysis
from churn_prediction import run_churn_model
from visualization import run_visualization

def main():
    print("===== STEP 1: RFM Analysis =====")
    run_rfm_analysis()
    print("===== STEP 2: Visualization =====")
    run_visualization()
    print("===== STEP 3: Churn Prediction =====")
    run_churn_model()
    print("All analysis finished!")

if __name__ == "__main__":
    main()