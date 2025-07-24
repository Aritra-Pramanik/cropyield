import pickle
import pandas as pd
import tkinter
import tkinter.messagebox
import customtkinter
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define the filename for the saved model
model_filename = 'knn_crop_yield_model.pkl'

# --- Load the Trained Model ---
loaded_model = None
try:
    if not os.path.exists(model_filename):
        tkinter.messagebox.showerror("Error", f"Model file '{model_filename}' not found.\nPlease make sure the model file is in the same directory as the script.")
        exit()

    print(f"Loading model from '{model_filename}'...")
    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully.")

except ImportError:
    tkinter.messagebox.showerror("Error", "Required libraries (pandas, scikit-learn, customtkinter, matplotlib) not found.\nPlease install them using: pip install pandas scikit-learn customtkinter matplotlib")
    exit()
except Exception as e:
    tkinter.messagebox.showerror("Error", f"An error occurred while loading the model: {e}")
    exit()

# --- GUI Application ---
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Crop Yield Predictor")
        self.geometry(f"{800}x{650}")  # Increased width to accommodate graph
        self.resizable(False, False)

        # Configure grid layout
        self.grid_columnconfigure(0, weight=0)  # Input column
        self.grid_columnconfigure(1, weight=1)  # Graph column
        self.grid_rowconfigure(0, weight=1)

        # --- Input Frame (Left Side) ---
        self.input_frame = customtkinter.CTkFrame(self)
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), weight=0)

        # Define features
        self.prediction_features = ['Crop', 'Crop_Year', 'Season', 'State', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        self.crop_options = [
            'Rice', 'Maize', 'Moong(Green Gram)', 'Urad', 'Groundnut', 'Sesamum',
            'Potato', 'Sugarcane', 'Wheat', 'Rapeseed &Mustard', 'Bajra', 'Jowar',
            'Arhar/Tur', 'Ragi', 'Gram', 'Small Millets', 'Cotton(Lint)', 'Onion',
            'Sunflower', 'Dry Chillies', 'Other Kharif Pulses', 'Horse-Gram',
            'Peas & Beans (Pulses)', 'Tobacco', 'Other Rabi Pulses', 'Soyabean',
            'Turmeric', 'Masoor', 'Ginger', 'Linseed', 'Castor Seed', 'Barley',
            'Sweet Potato', 'Garlic', 'Banana', 'Mesta', 'Tapioca', 'Coriander',
            'Niger Seed', 'Jute', 'Coconut', 'Safflower', 'Arecanut', 'Sannhamp',
            'Other Cereals', 'Cashewnut', 'Cowpea(Lobia)', 'Black Pepper',
            'Other Oilseeds', 'Moth', 'Khesari', 'Cardamom', 'Guar Seed',
            'Oilseeds Total', 'Other Summer Pulses'
        ]
        self.season_options = ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Autumn', 'Winter']
        self.state_options = [
            'Karnataka', 'Andhra Pradesh', 'West Bengal', 'Chhattisgarh', 'Bihar',
            'Madhya Pradesh', 'Uttar Pradesh', 'Tamil Nadu', 'Gujarat', 'Maharashtra',
            'Odisha', 'Assam', 'Uttarakhand', 'Nagaland', 'Puducherry', 'Meghalaya',
            'Jammu And Kashmir', 'Haryana', 'Himachal Pradesh', 'Kerala', 'Manipur',
            'Tripura', 'Mizoram', 'Punjab', 'Telangana', 'Arunachal Pradesh',
            'Jharkhand', 'Goa', 'Sikkim', 'Delhi'
        ]

        # Input widgets
        self.crop_label = customtkinter.CTkLabel(self.input_frame, text="Crop Type:")
        self.crop_label.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")
        self.crop_optionmenu = customtkinter.CTkOptionMenu(self.input_frame, values=self.crop_options)
        self.crop_optionmenu.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.crop_year_label = customtkinter.CTkLabel(self.input_frame, text="Crop Year:")
        self.crop_year_label.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")
        self.crop_year_entry = customtkinter.CTkEntry(self.input_frame)
        self.crop_year_entry.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.season_label = customtkinter.CTkLabel(self.input_frame, text="Season:")
        self.season_label.grid(row=4, column=0, padx=20, pady=(10, 5), sticky="w")
        self.season_optionmenu = customtkinter.CTkOptionMenu(self.input_frame, values=self.season_options)
        self.season_optionmenu.grid(row=5, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.state_label = customtkinter.CTkLabel(self.input_frame, text="State:")
        self.state_label.grid(row=6, column=0, padx=20, pady=(10, 5), sticky="w")
        self.state_optionmenu = customtkinter.CTkOptionMenu(self.input_frame, values=self.state_options)
        self.state_optionmenu.grid(row=7, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.annual_rainfall_label = customtkinter.CTkLabel(self.input_frame, text="Annual Rainfall (mm):")
        self.annual_rainfall_label.grid(row=8, column=0, padx=20, pady=(10, 5), sticky="w")
        self.annual_rainfall_entry = customtkinter.CTkEntry(self.input_frame)
        self.annual_rainfall_entry.grid(row=9, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.fertilizer_label = customtkinter.CTkLabel(self.input_frame, text="Fertilizer usage:")
        self.fertilizer_label.grid(row=10, column=0, padx=20, pady=(10, 5), sticky="w")
        self.fertilizer_entry = customtkinter.CTkEntry(self.input_frame)
        self.fertilizer_entry.grid(row=11, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.pesticide_label = customtkinter.CTkLabel(self.input_frame, text="Pesticide usage:")
        self.pesticide_label.grid(row=12, column=0, padx=20, pady=(10, 5), sticky="w")
        self.pesticide_entry = customtkinter.CTkEntry(self.input_frame)
        self.pesticide_entry.grid(row=13, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.predict_button = customtkinter.CTkButton(self.input_frame, text="Predict Yield", command=self.predict)
        self.predict_button.grid(row=14, column=0, padx=20, pady=20, sticky="ew")

        self.result_label = customtkinter.CTkLabel(self.input_frame, text="Predicted Crop Yield: --", font=customtkinter.CTkFont(size=16))
        self.result_label.grid(row=15, column=0, padx=20, pady=(0, 20))

        # --- Graph Frame (Right Side) ---
        self.graph_frame = customtkinter.CTkFrame(self)
        self.graph_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.graph_frame.grid_columnconfigure(0, weight=1)
        self.graph_frame.grid_rowconfigure(0, weight=1)

        # Placeholder label for graph
        self.placeholder_label = customtkinter.CTkLabel(self.graph_frame, text="Yield trend will appear here", font=customtkinter.CTkFont(size=16))
        self.placeholder_label.grid(row=0, column=0, padx=20, pady=20)

    def predict(self):
        """
        Retrieves input from GUI, makes prediction for the entered year, plots yield trend for the next 10 years,
        and annotates the entered year's yield on the graph.
        """
        if loaded_model is None:
            tkinter.messagebox.showerror("Model Error", "The prediction model was not loaded.")
            self.result_label.configure(text="Predicted Crop Yield: --")
            return

        try:
            # Get input values
            crop = self.crop_optionmenu.get()
            crop_year = int(self.crop_year_entry.get())
            season = self.season_optionmenu.get()
            state = self.state_optionmenu.get()
            annual_rainfall = float(self.annual_rainfall_entry.get())
            fertilizer = float(self.fertilizer_entry.get())
            pesticide = float(self.pesticide_entry.get())

            # --- Single Prediction for Entered Year ---
            new_data = pd.DataFrame([[crop, crop_year, season, state, annual_rainfall, fertilizer, pesticide]],
                                    columns=self.prediction_features)
            predicted_yield = loaded_model.predict(new_data)[0]
            self.result_label.configure(text=f"Predicted Crop Yield for {crop_year}: {predicted_yield:.2f}")

            # --- Predictions for Next 10 Years ---
            years = list(range(crop_year, crop_year + 10))
            yields = []
            for year in years:
                new_data = pd.DataFrame([[crop, year, season, state, annual_rainfall, fertilizer, pesticide]],
                                        columns=self.prediction_features)
                pred = loaded_model.predict(new_data)[0]
                yields.append(pred)

            # --- Plot the Yield Trend ---
            # Clear previous graph
            for widget in self.graph_frame.winfo_children():
                widget.destroy()

            # Create new plot with dark theme
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#2b2b2b')  # Dark background
            ax.set_facecolor('#2b2b2b')  # Dark plot background
            ax.plot(years, yields, marker='o', color='#00CC96', linewidth=2)  # Bright line color
            ax.set_title(f"Yield Trend for {crop} in {state}", color='white')
            ax.set_xlabel("Year", color='white')
            ax.set_ylabel("Predicted Yield", color='white')
            ax.grid(True, color='gray', linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', colors='white')

            # Annotate the entered year's yield
            ax.annotate(
                f'{predicted_yield:.2f}',
                xy=(crop_year, predicted_yield),
                xytext=(crop_year, predicted_yield + 0.1 * (max(yields) - min(yields))),  # Offset above point
                color='white',
                fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='white'),
                ha='center'
            )
            ax.scatter([crop_year], [predicted_yield], color='yellow', s=100, zorder=5)  # Highlight entered year

            plt.tight_layout()

            # Embed plot in GUI
            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        except ValueError:
            tkinter.messagebox.showerror("Input Error",
                                         "Please enter valid numerical values for Year, Rainfall, Fertilizer, and Pesticide.")
            self.result_label.configure(text="Predicted Crop Yield: --")
        except Exception as e:
            import traceback
            traceback.print_exc()
            tkinter.messagebox.showerror("Prediction Error",
                                         f"An error occurred during prediction: {e}\nCheck console for details.")
            self.result_label.configure(text="Predicted Crop Yield: --")

# --- Main execution ---
if __name__ == "__main__":
    customtkinter.set_appearance_mode("Dark")
    customtkinter.set_default_color_theme("blue")
    customtkinter.set_widget_scaling(1.0)
    app = App()
    app.mainloop()