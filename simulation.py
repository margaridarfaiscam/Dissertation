#%%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import simpy
import numpy as np
from datetime import datetime, timedelta
from joblib import load
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

# Load the trained models for each component
coxph_model_comp1 = load('best_coxph_model_comp1.pkl')
rsf_model_comp1 = load('best_rsf_model_comp1.pkl')
gbsa_model_comp1 = load('best_gbsa_model_comp1.pkl')
svm_model_comp1 = load('best_svm_model_comp1.pkl')

coxph_model_comp2 = load('best_coxph_model_comp2.pkl')
rsf_model_comp2 = load('best_rsf_model_comp2.pkl')
gbsa_model_comp2 = load('best_gbsa_model_comp2.pkl')
svm_model_comp2 = load('best_svm_model_comp2.pkl')

coxph_model_comp3 = load('best_coxph_model_comp3.pkl')
rsf_model_comp3 = load('best_rsf_model_comp3.pkl')
gbsa_model_comp3 = load('best_gbsa_model_comp3.pkl')
svm_model_comp3 = load('best_svm_model_comp3.pkl')

coxph_model_comp4 = load('best_coxph_model_comp4.pkl')
rsf_model_comp4 = load('best_rsf_model_comp4.pkl')
gbsa_model_comp4 = load('best_gbsa_model_comp4.pkl')
svm_model_comp4 = load('best_svm_model_comp4.pkl')

# Load and preprocess the dataset
wc_file = r'C:\Users\marga\Desktop\Dissertation\machine1_data_final.csv'
x_features_comp1 = ['volt', 'rotate', 'pressure', 'vibration', 'cycle_comp1']
x_features_comp2 = ['volt', 'rotate', 'pressure', 'vibration', 'cycle_comp2']
x_features_comp3 = ['volt', 'rotate', 'pressure', 'vibration', 'cycle_comp3']
x_features_comp4 = ['volt', 'rotate', 'pressure', 'vibration', 'cycle_comp4']
df_wc = pd.read_csv(wc_file)
df_wc['datetime'] = pd.to_datetime(df_wc['datetime'], format='%d/%m/%Y %H:%M')
df_wc['status_comp1'] = df_wc['replace_comp1'].map({True: False, False: True})
df_wc['ttf_comp1'] = df_wc['ttf_comp1'] + 1
df_wc['status_comp2'] = df_wc['replace_comp2'].map({True: False, False: True})
df_wc['ttf_comp2'] = df_wc['ttf_comp2'] + 1
df_wc['status_comp3'] = df_wc['replace_comp3'].map({True: False, False: True})
df_wc['ttf_comp3'] = df_wc['ttf_comp3'] + 1
df_wc['status_comp4'] = df_wc['replace_comp4'].map({True: False, False: True})
df_wc['ttf_comp4'] = df_wc['ttf_comp4'] + 1

# Feature scaling
scaler = StandardScaler()
df_wc[x_features_comp1] = scaler.fit_transform(df_wc[x_features_comp1])
df_wc[x_features_comp2] = scaler.fit_transform(df_wc[x_features_comp2])
df_wc[x_features_comp3] = scaler.fit_transform(df_wc[x_features_comp3])
df_wc[x_features_comp4] = scaler.fit_transform(df_wc[x_features_comp4])

# Calculate MTTF for both components
mttf_comp1 = df_wc['ttf_comp1'].mean()
mttf_comp2 = df_wc['ttf_comp2'].mean()
mttf_comp3 = df_wc['ttf_comp3'].mean()
mttf_comp4 = df_wc['ttf_comp4'].mean()
print(f"Calculated MTTF for comp1: {mttf_comp1} hours")
print(f"Calculated MTTF for comp2: {mttf_comp2} hours")
print(f"Calculated MTTF for comp3: {mttf_comp3} hours")
print(f"Calculated MTTF for comp4: {mttf_comp4} hours")

# Select a random sample of features for the simulation
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
machine_features_comp1 = df_wc.sample(n=1, random_state=RANDOM_SEED)[x_features_comp1].values[0]
machine_features_comp2 = df_wc.sample(n=1, random_state=RANDOM_SEED)[x_features_comp2].values[0]
machine_features_comp3 = df_wc.sample(n=1, random_state=RANDOM_SEED)[x_features_comp3].values[0]
machine_features_comp4 = df_wc.sample(n=1, random_state=RANDOM_SEED)[x_features_comp4].values[0]

# Streamlit UI
st.title("Machine Simulation Dashboard")

# Input date and simulation time
selected_date = st.date_input("Select a date", datetime(2015, 1, 1))
selected_time = st.time_input("Select a time", datetime(2015, 1, 1, 0, 0).time())
simulation_weeks = st.number_input("Select Time of Simulation (weeks)", min_value=1, max_value=52, value=4)

# Combine date and time into datetime
start_date = datetime.combine(selected_date, selected_time)

# Define the allowable start and end date range
allowed_start_date = datetime(2015, 1, 1, 6, 0)
allowed_end_date = datetime(2016, 1, 1, 6, 0)

# Display the simulation button
if st.button("Simulate"):
     # Validate the selected start date and time
    if not (allowed_start_date <= start_date <= allowed_end_date):
        st.error(f"Please select a start date and time between {allowed_start_date} and {allowed_end_date}.")
    else:
        # Define the simulation parameters
        REPAIR_TIME = 4      # Time it takes to repair the component in hours (12 hours)
        WEEKS = simulation_weeks  # Simulation time in weeks
        SIM_TIME = WEEKS * 7 * 24  # Simulation time in hours
        START_DATE = start_date  # Simulation start date
        REPAIR_COST = 40     # Cost to repair a component
        DISPLACEMENT_COST = 40  # Cost for displacement

        def convert_risk_to_time(risk_scores, training_data, component='comp1'):
            """Convert risk scores to predicted survival times."""
            median_risk = np.median(risk_scores)
            scale_factor = np.median(training_data[f'ttf_{component}']) / median_risk
            predicted_times = risk_scores * scale_factor
            return predicted_times

        def predict_time_to_failure(machine_features, model_choice, mttf, component='comp1'):
            """Predict the time to failure for a machine component based on its features."""
            features = np.array([machine_features])  # Ensure it is in the correct shape
            if component == 'comp1':
                if model_choice == 'coxph':
                    survival_function = coxph_model_comp1.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'rsf':
                    survival_function = rsf_model_comp1.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'gbsa':
                    survival_function = gbsa_model_comp1.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'svm':
                    risk_scores = svm_model_comp1.predict(features)
                    median_failure_time = convert_risk_to_time(risk_scores, df_wc, component='comp1')
                    median_failure_time = np.median(median_failure_time)  # Taking median for simplicity
                else:
                    median_failure_time = mttf_comp1  # Default to MTTF if no model is specified or available
            elif component == 'comp2':
                if model_choice == 'coxph':
                    survival_function = coxph_model_comp2.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'rsf':
                    survival_function = rsf_model_comp2.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'gbsa':
                    survival_function = gbsa_model_comp2.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'svm':
                    risk_scores = svm_model_comp2.predict(features)
                    median_failure_time = convert_risk_to_time(risk_scores, df_wc, component='comp2')
                    median_failure_time = np.median(median_failure_time)  # Taking median for simplicity
                else:
                    median_failure_time = mttf_comp2  # Default to MTTF if no model is specified or available
            elif component == 'comp3':
                if model_choice == 'coxph':
                    survival_function = coxph_model_comp3.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'rsf':
                    survival_function = rsf_model_comp3.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'gbsa':
                    survival_function = gbsa_model_comp3.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'svm':
                    risk_scores = svm_model_comp3.predict(features)
                    median_failure_time = convert_risk_to_time(risk_scores, df_wc, component='comp3')
                    median_failure_time = np.median(median_failure_time)  # Taking median for simplicity
                else:
                    median_failure_time = mttf_comp3  # Default to MTTF if no model is specified or available
            elif component == 'comp4':
                if model_choice == 'coxph':
                    survival_function = coxph_model_comp4.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'rsf':
                    survival_function = rsf_model_comp4.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'gbsa':
                    survival_function = gbsa_model_comp4.predict_survival_function(features)
                    median_failure_time = np.argmax(survival_function[0].y < 0.5)
                elif model_choice == 'svm':
                    risk_scores = svm_model_comp4.predict(features)
                    median_failure_time = convert_risk_to_time(risk_scores, df_wc, component='comp4')
                    median_failure_time = np.median(median_failure_time)  # Taking median for simplicity
                else:
                    median_failure_time = mttf_comp4  # Default to MTTF if no model is specified or available
            default_mttf = {
                'comp1': mttf_comp1,
                'comp2': mttf_comp2,
                'comp3': mttf_comp3,
                'comp4': mttf_comp4,
            }
            return median_failure_time if median_failure_time > 0 else default_mttf[component]

        class Machine:
            """A machine with components that may fail and need replacement."""
            def __init__(self, env, name, repairman, features_comp1, features_comp2, features_comp3, features_comp4, model_choice_comp1, model_choice_comp2, model_choice_comp3, model_choice_comp4, mttf_comp1, mttf_comp2, mttf_comp3, mttf_comp4):
                self.last_displacement_day = None
                self.repair_count_per_day = {}  # Track number of repairs per day
                self.env = env
                self.name = name
                self.repairman = repairman
                self.features_comp1 = features_comp1
                self.features_comp2 = features_comp2
                self.features_comp3 = features_comp3
                self.features_comp4 = features_comp4
                self.model_choice_comp1 = model_choice_comp1
                self.model_choice_comp2 = model_choice_comp2
                self.model_choice_comp3 = model_choice_comp3
                self.model_choice_comp4 = model_choice_comp4
                self.mttf_comp1 = mttf_comp1
                self.mttf_comp2 = mttf_comp2
                self.mttf_comp3 = mttf_comp3
                self.mttf_comp4 = mttf_comp4
                self.broken_dict = {comp: False for comp in ['comp1', 'comp2', 'comp3', 'comp4']}
                self.repairing_dict = {comp: False for comp in ['comp1', 'comp2', 'comp3', 'comp4']}
                self.remaining_time_dict = {
                    'comp1': self.predict_time_to_failure('comp1'),
                    'comp2': self.predict_time_to_failure('comp2'),
                    'comp3': self.predict_time_to_failure('comp3'),
                    'comp4': self.predict_time_to_failure('comp4')
                }
                self.failures_dict = {comp: 0 for comp in ['comp1', 'comp2', 'comp3', 'comp4']}
                self.total_cost = 0  # Total cost of repairs and displacement
                self.remaining_time_data = []
                self.repair_log = []  # List to store repair logs

                self.process = env.process(self.run_machine())
                self.update_process = env.process(self.update_remaining_times())  # Ensure this is only called once

            def log_event(self, component, event_type, cost=0):
                """Log events such as failures or repairs for a component."""
                event_datetime = START_DATE + timedelta(hours=int(self.env.now))
                event_message = f"{event_datetime}: {self.name} {component} {event_type}."
                print(event_message)
                
                self.remaining_time_data.append({
                    'datetime': event_datetime.strftime('%d/%m/%Y %H:%M'),
                    'event': f"{component} {event_type}",
                    'remaining_time_comp1': self.remaining_time_dict['comp1'],
                    'remaining_time_comp2': self.remaining_time_dict['comp2'],
                    'remaining_time_comp3': self.remaining_time_dict['comp3'],
                    'remaining_time_comp4': self.remaining_time_dict['comp4'],
                    'cost': cost
                })
                
                if event_type == 'repair started':
                    model_used = getattr(self, f'model_choice_{component}')
                    self.repair_log.append({
                        'component': component,
                        'start_time': event_datetime.strftime('%d/%m/%Y %H:%M'),
                        'model_used': model_used
                    })

            def predict_time_to_failure(self, component):
                features = getattr(self, f'features_{component}')
                model_choice = getattr(self, f'model_choice_{component}')
                mttf = getattr(self, f'mttf_{component}')
                predicted_time = predict_time_to_failure(features, model_choice, mttf, component)
                print(f"Predicted time to failure for {component}: {predicted_time}")
                return predicted_time

            def check_remaining_life(self):
                """Check if any component has less than 55 remaining life after repairs."""
                for comp, remaining_time in self.remaining_time_dict.items():
                    if remaining_time < 55:
                        current_datetime = START_DATE + timedelta(hours=int(self.env.now))
                        self.log_event(comp, f'has less than 55 remaining life ({remaining_time} hours)')

            def run_machine(self):
                """Run the machine as long as the simulation runs."""
                while True:
                    yield self.env.timeout(1)
                    for comp in self.broken_dict:
                        if not self.broken_dict[comp] and not self.repairing_dict[comp]:
                            self.remaining_time_dict[comp] -= 1
                            if self.remaining_time_dict[comp] <= 0:
                                self.remaining_time_dict[comp] = 0
                                self.log_event(comp, 'failed')
                                self.broken_dict[comp] = True
                                self.env.process(self.start_repair(comp))
                    # Check remaining life at the end of each day
                    if int(self.env.now) % 24 == 0:
                        self.check_remaining_life()

            def log_remaining_times(self):
                """Log the remaining times for each component."""
                current_datetime = START_DATE + timedelta(hours=int(self.env.now))
                self.remaining_time_data.append({
                    'datetime': current_datetime.strftime('%d/%m/%Y %H:%M'),
                    'remaining_time_comp1': self.remaining_time_dict['comp1'],
                    'remaining_time_comp2': self.remaining_time_dict['comp2'],
                    'remaining_time_comp3': self.remaining_time_dict['comp3'],
                    'remaining_time_comp4': self.remaining_time_dict['comp4']
                })
                print(f"{current_datetime}: (update process) Remaining times - Comp1: {self.remaining_time_dict['comp1']}, Comp2: {self.remaining_time_dict['comp2']}, Comp3: {self.remaining_time_dict['comp3']}, Comp4: {self.remaining_time_dict['comp4']}")

            def update_remaining_times(self):
                """Update the remaining times every 8 hours."""
                while True:
                    current_hour = int(self.env.now)
                    if current_hour % 8 == 0:
                        print(f"Debug: Calling log_remaining_times at hour {current_hour}")
                        self.log_remaining_times()
                    yield self.env.timeout(1)

            def start_repair(self, component):
                """Handles the repair process of a component."""
                if self.repairing_dict[component]:
                    return  # If the component is already being repaired, do nothing
                self.repairing_dict[component] = True
                self.failures_dict[component] += 1
                self.total_cost += REPAIR_COST  # Add the repair cost

                with self.repairman.request(priority=1) as req:
                    yield req
                    try:
                        repair_started = False
                        while not repair_started:
                            # Get the current simulation time
                            current_datetime = START_DATE + timedelta(hours=int(self.env.now))
                            print(f"Current time: {current_datetime}")
                            # Check if the current time is within the allowed repair window
                            if (current_datetime.weekday() < 5 and 8 <= current_datetime.hour < 17):
                                # Log the start time of the repair
                                if current_datetime.date() in self.repair_count_per_day:
                                    self.repair_count_per_day[current_datetime.date()] += 1
                                else:
                                    self.repair_count_per_day[current_datetime.date()] = 1
                                
                                if self.repair_count_per_day[current_datetime.date()] == 1:
                                    repair_time = REPAIR_TIME
                                else:
                                    repair_time = 1  # Subsequent repairs take 1 hour

                                # Add displacement cost if it's a new day or the first repair of the day
                                if self.last_displacement_day != current_datetime.date():
                                    self.total_cost += DISPLACEMENT_COST
                                    self.last_displacement_day = current_datetime.date()
                                    self.log_event(component, 'displacement cost added', cost=DISPLACEMENT_COST)

                                self.log_event(component, 'repair started', cost=REPAIR_COST)
                                yield self.env.timeout(repair_time)
                                repair_started = True
                            else:
                                # Calculate the next repair window start time
                                if current_datetime.weekday() >= 5:
                                    # If it's Saturday (5) or Sunday (6), calculate hours until next Monday 8 AM
                                    days_until_monday = (7 - current_datetime.weekday()) % 7
                                    next_repair_start = (days_until_monday * 24) + (8 - current_datetime.hour)
                                elif current_datetime.hour >= 17:
                                    # If it's a weekday but after 5 PM, wait until next day 8 AM
                                    next_repair_start = ((24 - current_datetime.hour) + 8)
                                else:
                                    # If it's a weekday but before 8 AM, wait until 8 AM
                                    next_repair_start = 8 - current_datetime.hour
                                print(f"Next repair start in: {next_repair_start} hours")
                                yield self.env.timeout(next_repair_start)
                    except simpy.Interrupt as interrupt:
                        print(f"{self.env.now}: Repair of {component} interrupted due to {interrupt.cause}")
                        self.env.process(self.start_repair(component))
                        return
                    self.complete_repair(component)
                
                self.repairing_dict[component] = False

            def complete_repair(self, component):
                """Completes the repair and updates the system state."""
                new_time = self.predict_time_to_failure(component)
                self.remaining_time_dict[component] = new_time
                self.broken_dict[component] = False
                self.log_event(component, 'repaired')
                print(f"Updated remaining time for {component}: {new_time}")

        # Setup and start the simulation
        print('Machine shop')
        env = simpy.Environment()
        repairman = simpy.PreemptiveResource(env, capacity=1)

        # Choose the models to use for comp1, comp2, comp3, and comp4
        model_choice_comp1 = 'rsf'  # Choose among 'coxph', 'rsf', 'gbsa', 'svm'
        model_choice_comp2 = 'gbsa'   # Choose among 'coxph', 'rsf', 'gbsa', 'svm'
        model_choice_comp3 = 'svm'  # Choose among 'coxph', 'rsf', 'gbsa', 'svm'
        model_choice_comp4 = 'svm'    # Choose among 'coxph', 'rsf', 'gbsa', 'svm''

        # Create a single machine with the chosen models for comp1, comp2, comp3, and comp4
        machine = Machine(env, 'Machine 1', repairman, machine_features_comp1, machine_features_comp2, machine_features_comp3, machine_features_comp4, model_choice_comp1, model_choice_comp2, model_choice_comp3, model_choice_comp4, mttf_comp1, mttf_comp2, mttf_comp3, mttf_comp4)
        env.run(until=SIM_TIME)


        ############################################################
        # Analysis/results
        print(f'Machine shop results after {WEEKS} weeks')
        print(f'{machine.name} used {model_choice_comp1} model for failure prediction of component 1.')
        print(f'{machine.name} component 1 failed {machine.failures_dict["comp1"]} times.')
        print(f'{machine.name} used {model_choice_comp2} model for failure prediction of component 2.')
        print(f'{machine.name} component 2 failed {machine.failures_dict["comp2"]} times.')
        print(f'{machine.name} used {model_choice_comp3} model for failure prediction of component 3.')
        print(f'{machine.name} component 3 failed {machine.failures_dict["comp3"]} times.')
        print(f'{machine.name} used {model_choice_comp4} model for failure prediction of component 4.')
        print(f'{machine.name} component 4 failed {machine.failures_dict["comp4"]} times.')
        print(f'Total cost of repairs and displacement: {machine.total_cost} €')

        # Write remaining time data to a CSV file
        df_remaining_time = pd.DataFrame(machine.remaining_time_data)
        output_csv_path = r'C:/Users/marga/Desktop/new_data_comparacao/sim_stream_23jul.csv'
        df_remaining_time.to_csv(output_csv_path, index=False)
        print(f"Remaining time data written to '{output_csv_path}'")

        ###

        # Analysis/results
        st.subheader("Simulation Report")
        st.write(f'Total cost of repairs and displacement: {machine.total_cost} €')

        failures = {
            'comp1': machine.failures_dict['comp1'],
            'comp2': machine.failures_dict['comp2'],
            'comp3': machine.failures_dict['comp3'],
            'comp4': machine.failures_dict['comp4'],
        }

        # Calculate end date based on the simulation time
        end_date = start_date + timedelta(hours=SIM_TIME)
        st.write(f'Simulation start date: {start_date}')
        st.write(f'Simulation end date: {end_date}')

        # Convert start_date and end_date to Matplotlib date format
        start_date_num = mdates.date2num(start_date)
        end_date_num = mdates.date2num(end_date)

        # Display repair log
        st.subheader("Repair Log")
        repair_log_df = pd.DataFrame(machine.repair_log)
        repair_log_df.index = repair_log_df.index + 1
        st.write(repair_log_df)

        # Plotting Remaining Useful Life
        df_remaining_time = pd.DataFrame(machine.remaining_time_data)
        df_remaining_time['datetime'] = pd.to_datetime(df_remaining_time['datetime'], format='%d/%m/%Y %H:%M')  # Ensure datetime column is in datetime format

        fig, ax = plt.subplots()
        for comp in ['remaining_time_comp1', 'remaining_time_comp2', 'remaining_time_comp3', 'remaining_time_comp4']:
            ax.plot(df_remaining_time['datetime'], df_remaining_time[comp], label=comp)

        # Improve x-axis visibility
        ax.set_xlim([start_date_num, end_date_num])

        # Set minor ticks for every day
        ax.xaxis.set_minor_locator(mdates.DayLocator())

        # Create custom major ticks starting from the first day and then every 7 days
        major_ticks = [start_date + timedelta(days=i*7) for i in range(int((end_date - start_date).days / 7) + 1)]
        ax.set_xticks(major_ticks)

        # Set major tick labels
        ax.set_xticklabels([date.strftime('%d.%m.%Y') for date in major_ticks])

        plt.xticks(rotation=45, fontsize=8)  # Set the font size for x-axis labels
        plt.yticks(fontsize=8)  # Set the font size for y-axis labels
        ax.legend(fontsize=8)  # Set the font size for the legend
        ax.set_xlabel("Date", fontsize=10)  # Optionally set a smaller font size for x-axis label
        ax.set_ylabel("Remaining Time (hours)", fontsize=10)  # Optionally set a smaller font size for y-axis label
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines for better visibility
        st.pyplot(fig)

        # Plotting Component Failures
        fig, ax = plt.subplots()
        bars = ax.bar(failures.keys(), failures.values())
        plt.xticks(rotation=45, fontsize=8)  # Set the font size for x-axis labels
        plt.yticks(fontsize=8)  # Set the font size for y-axis labels
        ax.set_xlabel("Component", fontsize=10)  # Optionally set a smaller font size for x-axis label
        ax.set_ylabel("Failures", fontsize=10)  # Optionally set a smaller font size for y-axis label

        # Adding labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=8)

        st.pyplot(fig)
