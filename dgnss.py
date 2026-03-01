import pandas as pd
Base_True_Lat = 12.990123
Base_True_Lon = 80.223452
df = pd.read_csv("dgps_rover_data.csv")
df["delta_lat"] = Base_True_Lat - df["Base Latitude (Measured)"]
df["delta_lon"] = Base_True_Lon - df["Base Longitude (Measured)"]
df["Rover Latitude (DGNSS)"] = (
    df["Rover Latitude (Measured)"] + df["delta_lat"]
)
df["Rover Longitude (DGNSS)"] = (
    df["Rover Longitude (Measured)"] + df["delta_lon"]
)
output_file = "dgnss_corrected_output.csv"
df.to_csv(output_file, index=False)
print("New file saved as:", output_file)