import abc

import pandas as pd

from typing import List, Dict, Any

class Feature(abc.ABC):

    name: str = "Base Feature"
    description: str = "Base feature class"

    @abc.abstractmethod
    def apply(self, df: pd.DataFrame) -> None:
        pass

class FeatureClass(abc.ABC):

    name: str = "Base Feature Class"
    feature_list: List[Feature] = list()

    @abc.abstractmethod
    def apply(self, df: pd.DataFrame) -> None:
        pass



class ElectricalResistance(Feature):

    name = "Electrical Resistance"
    description = "Deviations in calculated electrical resistance from expected values can indicate anomalies such as motor winding degradation, short circuits, or changes in mechanical load that affect current draw for a given voltage."

    def apply(self, df: pd.DataFrame) -> None:
        if "Voltage" in df.columns and "Current" in df.columns:
            df["Electrical_Resistence"] = df["Voltage"] / df["Current"]
        else:
            raise ValueError("""DataFrame must contain "Voltage" and "Current" columns.""")

class FlowResistance(Feature):

    name = "Flow Resistance"
    description = "Increase in flow resistance for a given flow rate or pressure can indicate blockages, increased friction due to scaling or debris, or issues with the pump itself, thus signaling an anomaly in the system."

    def apply(self, df: pd.DataFrame) -> None:
        if "Pressure" in df.columns and "Volume Flow RateRMS" in df.columns:
            df["Flow_Resistance"] = df["Pressure"] / (df["Volume Flow RateRMS"] + 1e-6)
        else:
            raise ValueError("""DataFrame must contain "Pressure" and "Volume Flow RateRMS" columns.""")
        
class OperationalTemperatureRatio(Feature):

    name = "Operational Temperature Ratio"
    description = "Significant deviations in this metric from a normal range can indicate issues such as inefficient heat transfer, motor overheating, or abnormal fluid heating due to cavitation, thereby signaling an anomaly."

    def apply(self, df: pd.DataFrame) -> None:
        if "Temperature" in df.columns and "Thermocouple" in df.columns:
            df["Temperature_EbyF"] = df["Temperature"] / (df["Thermocouple"] + 1e-6)
        else:
            raise ValueError("""DataFrame must contain "Temperature" and "Thermocouple" columns.""")
        
class OperationalVibration(Feature):

    name = "Operational Vibration"
    description = "Increase in Operational Vibration beyond normal thresholds can directly indicate mechanical issues such as imbalance, misalignment, bearing wear, or cavitation within the pump or motor, thus serving as a strong anomaly indicator."

    def apply(self, df: pd.DataFrame) -> None:
        if "Accelerometer1RMS" in df.columns and "Accelerometer2RMS" in df.columns:
            df["AccelerometerRMS"] = df["Accelerometer1RMS"] * df["Accelerometer2RMS"]
        else:
            raise ValueError("""DataFrame must contain "Accelerometer1RMS" and "Accelerometer2RMS" columns.""")
        
class SpecificPower(Feature):

    name = "Specific Power"
    description = "Increase in Operational Vibration beyond normal thresholds can directly indicate mechanical issues such as imbalance, misalignment, bearing wear, or cavitation within the pump or motor, thus serving as a strong anomaly indicator."

    def apply(self, df: pd.DataFrame) -> None:
        if "Voltage" in df.columns and "Current" in df.columns and "Volume Flow RateRMS" in df.columns:
            df["Specific_Power"] = (df["Voltage"] * df["Current"]) / df["Volume Flow RateRMS"]
        else:
            raise ValueError("""DataFrame must contain "Voltage", "Current" and "Volume Flow RateRMS" columns.""")

class InteractionFeatures(FeatureClass):

    name = "Interaction Features"
    feature_list = [
                        ElectricalResistance(), 
                        FlowResistance(),
                        OperationalTemperatureRatio(),
                        OperationalVibration(),
                        SpecificPower(),
    ]

    def apply(self, df: pd.DataFrame) -> None:
        for feature in self.feature_list:
            feature.apply(df)


        
class VibrationalBalance(Feature):

    name = "Vibrational Balance"
    description = "Increase in Vibrational Balance can indicate an imbalance in the system's vibration, potentially pointing to issues like misalignment or uneven bearing wear."

    def apply(self, df: pd.DataFrame) -> None:
        if "Accelerometer1RMS" in df.columns and "Accelerometer2RMS" in df.columns:
            df["Accelerometer_Balance"] = df["Accelerometer2RMS"] - df["Accelerometer1RMS"]
        else:
            raise ValueError("""DataFrame must contain "Accelerometer1RMS" and "Accelerometer2RMS" columns.""")
        
class TemperatureBalance(Feature):

    name = "Temperature Balance"
    description = "Change in Temperature Balance from its normal operating range can signal issues like inefficient cooling of the motor, excessive heat generation from the fluid (e.g., due to cavitation), or an external heat imbalance, thereby indicating an anomaly."

    def apply(self, df: pd.DataFrame) -> None:
        if "Temperature" in df.columns and "Thermocouple" in df.columns:
            df["Temperature_Difference"] = df["Temperature"] - df["Thermocouple"]
        else:
            raise ValueError("""DataFrame must contain "Temperature" and "Thermocouple" columns.""")
        
class DifferenceFeatures(FeatureClass):

    name = "Difference Features"
    feature_list = [
                        VibrationalBalance(),
                        TemperatureBalance()
    ]

    def apply(self, df: pd.DataFrame) -> None:
        for feature in self.feature_list:
            feature.apply(df)


        
class HydraulicPower(Feature):

    name = "Hydraulic Power"
    description = "Drop in Hydraulic Power relative to the electrical power input indicates a loss of pump efficiency or internal issues, serving as a strong indicator of an anomaly."

    def apply(self, df: pd.DataFrame) -> None:
        if "Volume Flow RateRMS" in df.columns and "Pressure" in df.columns:

            density_of_water = 1000 # kg/m^3
            acceleration_due_to_gravity = 9.81 # m/s^2
            Q_factor = 1.66667e-5 # m^3/s
            specific_gravity_of_water = 1

            df["Hydraulic_Power"] = (Q_factor*df["Volume Flow RateRMS"])*(density_of_water * acceleration_due_to_gravity * (df["Pressure"] / (0.0981 * specific_gravity_of_water)))
        else:
            raise ValueError("""DataFrame must contain "Volume Flow RateRMS" and "Pressure" columns.""")
        
class ElectricalPower(Feature):

    name = "Electrical Power"
    description = "Unexpected increases or fluctuations in Electrical Power, without corresponding changes in pump output, can indicate issues like motor overloading, winding problems, or increased mechanical resistance in the pump, thus signaling an anomaly."

    def apply(self, df: pd.DataFrame) -> None:
        if "Voltage" in df.columns and "Current" in df.columns:

            power_factor = 0.85

            df["Electrical_Power"] = (df["Voltage"] * df["Current"])* power_factor
        else:
            raise ValueError("""DataFrame must contain "Voltage" and "Current" columns.""")
        
class PumpEfficiency(Feature):

    name = "Pump Efficiency"
    description = "Decrease in Pump Efficiency for a given operating condition directly signals degradation, internal wear, cavitation, or blockages within the pump, making it a highly effective anomaly detector."

    def apply(self, df: pd.DataFrame) -> None:
        if "Voltage" in df.columns and "Current" in df.columns and "Volume Flow RateRMS" in df.columns and "Pressure" in df.columns:

            density_of_water = 1000 # kg/m^3
            acceleration_due_to_gravity = 9.81 # m/s^2
            Q_factor = 1.66667e-5 # m^3/s
            specific_gravity_of_water = 1
            hydraulic_power = (Q_factor*df["Volume Flow RateRMS"])*(density_of_water * acceleration_due_to_gravity * (df["Pressure"] / (0.0981 * specific_gravity_of_water)))

            power_factor = 0.85
            electrical_power = (df["Voltage"] * df["Current"]) * power_factor

            df["Actual_Efficiency"] = hydraulic_power / electrical_power
        else:
            raise ValueError("""DataFrame must contain "Voltage", "Volume Flow RateRMS", "Pressure" and "Current" columns.""")
        
class EfficiencyPerformanceMetrics(FeatureClass):

    name = "Efficiency & Performance Metrics"
    feature_list = [
                        HydraulicPower(),
                        ElectricalPower(),
                        PumpEfficiency(),
    ]

    def apply(self, df: pd.DataFrame) -> None:
        for feature in self.feature_list:
            feature.apply(df)

        
class VibrationIntensityRatio(Feature):

    name = "Vibration Intensity Ratio"
    description = "Increase in Vibration per Unit Flow can occur when the flow rate is not proportionally high, it can point to issues like cavitation, internal impeller damage, or bearing problems not directly related to the hydraulic load, thus signaling an anomaly."

    def apply(self, df: pd.DataFrame) -> None:
        if "Accelerometer1RMS" in df.columns and "Accelerometer2RMS" in df.columns:
            df["Vibration_Intensity_Ratio"] = df["Accelerometer1RMS"] / df["Accelerometer2RMS"]
        else:
            raise ValueError("""DataFrame must contain "Accelerometer1RMS" and "Accelerometer2RMS" columns.""")
        
class VibrationPerUnitFlow(Feature):

    name = "Vibration per Unit Flow"
    description = "Increase in Vibration per Unit Flow can occur when the flow rate is not proportionally high, it can point to issues like cavitation, internal impeller damage, or bearing problems not directly related to the hydraulic load, thus signaling an anomaly."

    def apply(self, df: pd.DataFrame) -> None:
        if "Accelerometer1RMS" in df.columns and "Accelerometer2RMS" in df.columns and "Volume Flow RateRMS" in df.columns:
            df["AccelerometerRMS_pu_Pressure"] = (df["Accelerometer1RMS"] * df["Accelerometer2RMS"]) / (df["Volume Flow RateRMS"] + 1e-6)
        else:
            raise ValueError("""DataFrame must contain "Accelerometer1RMS", "Accelerometer2RMS" and "Volume Flow RateRMS" columns.""")
        
class EngineTemperaturePerUnitPower(Feature):

    name = "Engine Temperature per Unit Power"
    description = "Unusual increase in Engine Temperature Per Unit Electrical Power suggests issues such as motor winding problems, cooling system failure, or increased internal friction, thereby signaling an anomaly."

    def apply(self, df: pd.DataFrame) -> None:
        if "Voltage" in df.columns and "Current" in df.columns and "Temperature" in df.columns:

            power_factor = 0.85
            electrical_power = (df["Voltage"] * df["Current"]) * power_factor

            df["Engine_Temperature_pu_Power"] = df["Temperature"] / electrical_power
        else:
            raise ValueError("""DataFrame must contain "Voltage", "Temperature" and "Current" columns.""")
        
class EfficiencyTemperatureBalanceRatio(Feature):

    name = "Efficiency & Temperature Balance Ratio"
    description = "A disproportionate relationship between a drop in efficiency and the motor/fluid temperature balance can help distinguish between hydraulic problems (e.g., worn impeller) and electrical/motor-related issues, thus aiding in anomaly detection."

    def apply(self, df: pd.DataFrame) -> None:
        if "Voltage" in df.columns and "Current" in df.columns and "Temperature" in df.columns and "Thermocouple" in df.columns and "Volume Flow RateRMS" in df.columns and "Pressure" in df.columns:

            density_of_water = 1000 # kg/m^3
            acceleration_due_to_gravity = 9.81 # m/s^2
            Q_factor = 1.66667e-5 # m^3/s
            specific_gravity_of_water = 1
            hydraulic_power = (Q_factor*df["Volume Flow RateRMS"])*(density_of_water * acceleration_due_to_gravity * (df["Pressure"] / (0.0981 * specific_gravity_of_water)))

            power_factor = 0.85
            electrical_power = (df["Voltage"] * df["Current"]) * power_factor

            temperature_difference = df["Temperature"] - df["Thermocouple"]

            df["Efficiency_Temperature_Difference_Ratio"] = (hydraulic_power / electrical_power) / temperature_difference
        else:
            raise ValueError("""DataFrame must contain "Voltage", "Temperature" and "Current" columns.""")
        
class HigherOrderInteractiveFeatures(FeatureClass):

    name = "Higher Order Interactive Features"
    feature_list = [
                        VibrationIntensityRatio(),
                        VibrationPerUnitFlow(),
                        EngineTemperaturePerUnitPower(),
                        EfficiencyTemperatureBalanceRatio(),
    ]

    def apply(self, df: pd.DataFrame) -> None:
        for feature in self.feature_list:
            feature.apply(df)



def apply_feature_engineering(selected_features: List[Feature], dataframe: pd.DataFrame) -> None:
    for feature in selected_features:
        feature.apply(dataframe)