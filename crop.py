import pickle
import pandas as pd
import tkinter
import tkinter.messagebox
import customtkinter
import os

# Define the filename for the saved model
model_filename = 'knn_crop_yield_model.pkl'

# --- Load the Trained Model ---
loaded_model = None
try:
    # Check if the model file exists
    if not os.path.exists(model_filename):
        tkinter.messagebox.showerror("Error", f"Model file '{model_filename}' not found.\nPlease make sure the model file is in the same directory as the script.")
        exit()

    print(f"Loading model from '{model_filename}'...")
    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully.")

except ImportError:
    tkinter.messagebox.showerror("Error", "Required libraries (pandas, scikit-learn, customtkinter) not found.\nPlease install them using: pip install pandas scikit-learn customtkinter")
    exit()
except Exception as e:
    tkinter.messagebox.showerror("Error", f"An error occurred while loading the model: {e}")
    exit()

# --- GUI Application ---

# Set default appearance and scaling before creating the main window
customtkinter.set_appearance_mode("Dark")  # Set default to Dark mode
customtkinter.set_default_color_theme("blue") # You can keep or change the theme
customtkinter.set_widget_scaling(1.0) # Set default to 100% scaling


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Crop Yield Predictor")
        self.geometry(f"{500}x{650}") # Adjusted geometry again to fit all inputs
        self.resizable(False, False) # Prevent resizing for simplicity

        # configure grid layout (single column for inputs)
        self.grid_columnconfigure(0, weight=1) # Make the single column expandable
        # Configure rows for input widgets, button, and result
        # We need a row for each label and each input widget, plus the button and result
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), weight=0)
        self.grid_rowconfigure(16, weight=1) # Make the last row expandable for spacing


        # --- Input Widgets for Prediction ---

        # Define the features that the model expects
        self.prediction_features = ['Crop', 'Crop_Year', 'Season', 'State', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

        # Get unique values for dropdowns (Placeholder - replace with loading from your data)
        # In a real application, load these from your original 'df' or a separate file.
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

        # Labels and Entry/OptionMenu widgets for prediction inputs
        # Placing widgets directly in the main window, in the single column (column 0)

        self.crop_label = customtkinter.CTkLabel(self, text="Crop Type:")
        self.crop_label.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")
        self.crop_optionmenu = customtkinter.CTkOptionMenu(self, values=self.crop_options)
        self.crop_optionmenu.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")


        self.crop_year_label = customtkinter.CTkLabel(self, text="Crop Year:")
        self.crop_year_label.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")
        self.crop_year_entry = customtkinter.CTkEntry(self)
        self.crop_year_entry.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")


        self.season_label = customtkinter.CTkLabel(self, text="Season:")
        self.season_label.grid(row=4, column=0, padx=20, pady=(10, 5), sticky="w")
        self.season_optionmenu = customtkinter.CTkOptionMenu(self, values=self.season_options)
        self.season_optionmenu.grid(row=5, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.state_label = customtkinter.CTkLabel(self, text="State:")
        self.state_label.grid(row=6, column=0, padx=20, pady=(10, 5), sticky="w")
        self.state_optionmenu = customtkinter.CTkOptionMenu(self, values=self.state_options)
        self.state_optionmenu.grid(row=7, column=0, padx=20, pady=(0, 10), sticky="ew")

        # --- Corrected: Added Entry for Annual Rainfall ---
        self.annual_rainfall_label = customtkinter.CTkLabel(self, text="Annual Rainfall (mm):")
        self.annual_rainfall_label.grid(row=8, column=0, padx=20, pady=(10, 5), sticky="w")
        self.annual_rainfall_entry = customtkinter.CTkEntry(self)
        self.annual_rainfall_entry.grid(row=9, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.fertilizer_label = customtkinter.CTkLabel(self, text="Fertilizer usage:")
        self.fertilizer_label.grid(row=10, column=0, padx=20, pady=(10, 5), sticky="w")
        self.fertilizer_entry = customtkinter.CTkEntry(self)
        self.fertilizer_entry.grid(row=11, column=0, padx=20, pady=(0, 10), sticky="ew")

        # --- Corrected: Added Entry for Pesticide Usage ---
        self.pesticide_label = customtkinter.CTkLabel(self, text="Pesticide usage:")
        self.pesticide_label.grid(row=12, column=0, padx=20, pady=(10, 5), sticky="w")
        self.pesticide_entry = customtkinter.CTkEntry(self)
        self.pesticide_entry.grid(row=13, column=0, padx=20, pady=(0, 10), sticky="ew")


        # Prediction Button
        self.predict_button = customtkinter.CTkButton(self, text="Predict Yield", command=self.predict)
        self.predict_button.grid(row=14, column=0, padx=20, pady=20, sticky="ew")

        # --- Corrected: Result Label for displaying Output ---
        self.result_label = customtkinter.CTkLabel(self, text="Predicted Crop Yield: --", font=customtkinter.CTkFont(size=16)) # Initial text
        self.result_label.grid(row=15, column=0, padx=20, pady=(0, 20))

    def predict(self):
        """
        Retrieves input from GUI, makes prediction, and displays result in a dialog box.
        """
        if loaded_model is None:
            tkinter.messagebox.showerror("Model Error", "The prediction model was not loaded.")
            self.result_label.configure(text="Predicted Crop Yield: --")  # Reset label
            return

        try:
            # Get input values from the GUI widgets
            crop = self.crop_optionmenu.get()
            crop_year = int(self.crop_year_entry.get())
            season = self.season_optionmenu.get()
            state = self.state_optionmenu.get()
            annual_rainfall = float(self.annual_rainfall_entry.get())
            fertilizer = float(self.fertilizer_entry.get())
            pesticide = float(self.pesticide_entry.get())

            # Define features (must match training features) - important for the DataFrame
            features = ['Crop', 'Crop_Year', 'Season', 'State', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

            # Create a pandas DataFrame for the new data
            new_data = pd.DataFrame([[crop, crop_year, season, state, annual_rainfall, fertilizer, pesticide]],
                                    columns=features)

            # Make prediction using the loaded model
            predicted_yield = loaded_model.predict(new_data)

            # --- Display the result in a dialog box ---
            # We can still update the label in the main window if desired,
            # but the primary output will be the dialog.
            self.result_label.configure(text="Predicted Crop Yield: --")  # Optional: reset label or show processing
            tkinter.messagebox.showinfo("Prediction Result", f"The predicted crop yield is: {predicted_yield[0]:.2f}")

        except ValueError:
            tkinter.messagebox.showerror("Input Error",
                                         "Please enter valid numerical values for Year, Rainfall, Fertilizer, and Pesticide.")
            self.result_label.configure(text="Predicted Crop Yield: --")  # Reset result label on error
        except Exception as e:
            # Print the full traceback to the console for debugging
            import traceback
            traceback.print_exc()
            tkinter.messagebox.showerror("Prediction Error",
                                         f"An error occurred during prediction: {e}\nCheck console for details.")
            self.result_label.configure(text="Predicted Crop Yield: --")  # Reset result label on error


# --- Main execution ---
if __name__ == "__main__":
    app = App()
    app.mainloop()