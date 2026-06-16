import csv

import pandas as pd
import numpy as np
import gtsam
from gtsam import symbol, Pose3, Rot3, Point3
import matplotlib.pyplot as plt

df = pd.read_csv('1.05_datas.csv')

signals = {
    '/vectornav/imu.linear_acceleration.x': 'acc_x',
    '/vectornav/imu.linear_acceleration.y': 'acc_y',
    '/vectornav/imu.linear_acceleration.z': 'acc_z',
    '/vectornav/imu.angular_velocity.x': 'gyro_x',
    '/vectornav/imu.angular_velocity.y': 'gyro_y',
    '/vectornav/imu.angular_velocity.z': 'gyro_z',
    '/putm_vcl/amk/rear/right/actual_values1.actual_velocity': 'mot_rr',
    '/putm_vcl/amk/rear/left/actual_values1.actual_velocity': 'mot_rl',
    '/putm_vcl/amk/front/right/actual_values1.actual_velocity': 'mot_fr',
    '/putm_vcl/amk/front/left/actual_values1.actual_velocity': 'mot_fl'
}

master_clock = df[df['topic'] == '/vectornav/imu.linear_acceleration.x'][['elapsed time']].sort_values('elapsed time').drop_duplicates()

# Synchronizacja


wheel_advance = 1.05 

synced_df = master_clock.copy()

for topic, name in signals.items():
    sensor_data = df[df['topic'] == topic][['elapsed time', 'value']].sort_values('elapsed time').copy()
    
    # Przyspieszamy sygnał silników
    if 'mot' in name:
        sensor_data['elapsed time'] = sensor_data['elapsed time'] - wheel_advance
    
    # Synchronizacja
    synced_df = pd.merge_asof(synced_df, 
                             sensor_data.rename(columns={'value': name}), 
                             on='elapsed time', 
                             direction='backward')

synced_df.fillna(0, inplace=True)


# KONWERSJA DO NUMPY
data_matrix = synced_df.to_numpy()
gear = 14.25
r=0.198
motors_to_velocity = (2 * np.pi * r) / (gear * 60)  
data_matrix[:,7] *=-1
data_matrix[:,10] *=-1
data_matrix[:, 7:11] *= motors_to_velocity  
buf = data_matrix[:, 1]
data_matrix[:, 1] = data_matrix[:, 2]
data_matrix[:, 2] = buf
data_matrix[:, 1] *= -1  # Odwracamy oś X, jeśli jest skierowana w dół

t = data_matrix[:, 0]
acc = data_matrix[:, 1:4]     # Kolumny 1, 2, 3 (acc_x, y, z)
gyro = data_matrix[:, 4:7]    # Kolumny 4, 5, 6 (gyro_x, y, z)
motors = data_matrix[:, 7:]   # Wszystkie kolumny od 7 do końca (silniki)

print(f"Kształt macierzy: {data_matrix.shape}") 
print("Przykładowy wiersz (t, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, rr, rl, fr, fl):")
print(data_matrix[0:3])

t = data_matrix[:, 0]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 1. Przyspieszenia (IMU)
ax1.plot(t, data_matrix[:, 1], label='Acc X')
ax1.plot(t, data_matrix[:, 2], label='Acc Y')
ax1.plot(t, data_matrix[:, 3], label='Acc Z')
ax1.set_ylabel('Przyspieszenie [m/s^2]')
ax1.set_title('Dane z IMU i Silników (Zsynchronizowane)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Prędkości kątowe (IMU)
ax2.plot(t, data_matrix[:, 4], label='Gyro X')
ax2.plot(t, data_matrix[:, 5], label='Gyro Y')
ax2.plot(t, data_matrix[:, 6], label='Gyro Z', color='black')
ax2.set_ylabel('Prędkość kątowa [rad/s]')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. Prędkości silników (AMK)
ax3.plot(t, data_matrix[:, 7], label='RR')
ax3.plot(t, data_matrix[:, 8], label='RL')
ax3.plot(t, data_matrix[:, 9], label='FR')
ax3.plot(t, data_matrix[:, 10], label='FL')
ax3.set_ylabel('Prędkość silnika [RPM/raw]')
ax3.set_xlabel('Czas [s]')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

print(data_matrix.shape)  

plt.tight_layout()
plt.show()

data_matrix_downsampled = data_matrix[::2, :].copy()


t_half = data_matrix_downsampled[:, 0]
motors_half = data_matrix_downsampled[:, 7:] # Tylko kolumny silników

print(f"Oryginalna liczba próbek: {len(data_matrix)}")
print(f"Liczba próbek po redukcji: {len(data_matrix_downsampled)}")


motors_matrix_reduced = np.column_stack((
    data_matrix[::2, 0],  # Co drugi timestamp
    data_matrix[::2, 7:]  # Co drugi wynik z silników
))
print(motors_matrix_reduced[0:4])
print("Kształt macierzy silników (co drugi wynik):", motors_matrix_reduced.shape)
  
initial_rotation = gtsam.Rot3()  

g= 9.81

params = gtsam.PreintegrationParams.MakeSharedU(g)

static_accel_bias = np.array([0.5, -0.025, 0.008])
static_gyro_bias = np.array([0.0005, 0.00066, 0.00176])
#biasy dla przyspieszenia i prędkości kątowej
params.setAccelerometerCovariance(np.eye(3) * 0.08**2)
params.setGyroscopeCovariance(np.eye(3) * 0.005**2)
params.setIntegrationCovariance(np.eye(3) * 1e-2)

initial_rotation = gtsam.Rot3.Quaternion(-0.6964, 0.0151, -0.01794, 0.71635)

bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6) 

wheel_velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.001)

isam = gtsam.ISAM2(gtsam.ISAM2Params())
graph = gtsam.NonlinearFactorGraph()
initial_values = gtsam.Values()


initial_pose = Pose3(initial_rotation, Point3(0, 0, 0))

#do obliczenia odchylenie standardowe dla accelerometru, prędkości kątowej


initial_bias = gtsam.imuBias.ConstantBias(static_accel_bias, static_gyro_bias)


X = lambda i: symbol('x', i)
V = lambda i: symbol('v', i)
B = lambda i: symbol('b', i)

# Wstawiamy tę pozę do wartości początkowych grafu
v_start = np.array([0.0, 0.0, 0.0])



initial_values.insert(X(0), initial_pose)
initial_values.insert(V(0), v_start)
initial_values.insert(B(0), initial_bias)

graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.01)))
graph.add(gtsam.PriorFactorVector(V(0), v_start, gtsam.noiseModel.Isotropic.Sigma(3, 0.01)))
graph.add(gtsam.PriorFactorConstantBias(B(0), initial_bias, gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)))



result = initial_values
preint = gtsam.PreintegratedImuMeasurements(params, initial_bias)
state_idx = 0

estimated_v = []
estimated_t = []

for i in range(1, len(acc)):
    dt = t[i] - t[i-1]
    preint.integrateMeasurement(acc[i], gyro[i], dt)
    if i % 2 == 0:
        new_idx = state_idx + 1
        graph.add(gtsam.ImuFactor(X(state_idx), V(state_idx), X(new_idx), V(new_idx), B(state_idx), preint))
        graph.add(gtsam.BetweenFactorConstantBias(B(state_idx), B(new_idx), gtsam.imuBias.ConstantBias(), bias_noise))
        mean_velocity = (motors_matrix_reduced[state_idx, 1] + motors_matrix_reduced[state_idx, 2])/2  # Średnia prędkość z silników
        v_measured = np.array([mean_velocity, 0.0, 0.0])
        graph.add(gtsam.PriorFactorVector(V(new_idx), v_measured, wheel_velocity_noise))
        last_nav_state = gtsam.NavState(result.atPose3(X(state_idx)), result.atVector(V(state_idx)))
        predicted_state = preint.predict(last_nav_state, result.atConstantBias(B(state_idx)))
    
        initial_values.insert(X(new_idx), predicted_state.pose())
        initial_values.insert(V(new_idx), predicted_state.velocity())
        initial_values.insert(B(new_idx), result.atConstantBias(B(state_idx)))

        isam.update(graph, initial_values)
        result = isam.calculateEstimate()
    
        graph = gtsam.NonlinearFactorGraph()
        initial_values.clear()
    
        # resetowanie preintegracji IMU do czystego stanu
        preint.resetIntegrationAndSetBias(result.atConstantBias(B(new_idx)))
        state_idx = new_idx
        current_v = result.atVector(V(new_idx))
        estimated_v.append(current_v)
        estimated_t.append(t[i])
        print(f"Estimation: {i}")
est_t = np.array(estimated_t)
est_v = np.array(estimated_v)[:, 0]

df_imu = pd.read_csv('velocity1_05.csv')
target_topic = '/vectornav/velocity_body.twist.twist.linear.x'
imu_raw = df_imu[df_imu['topic'] == target_topic]

t_imu = imu_raw['elapsed time'].values
v_imu = imu_raw['value'].values * -1  # Odwracamy znak referencji


v_imu_synced = np.interp(est_t, t_imu, v_imu)

output_filename = "porownanie_predkosci_final.csv"

with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Czas [s]', 'Vx_GTSAM [m/s]', 'Vx_IMU_Ref [m/s]'])
    
    for i in range(len(est_t)):
        writer.writerow([est_t[i], est_v[i], v_imu_synced[i]])

print(f"Zapisano zsynchronizowane dane do: {output_filename}")

plt.figure(figsize=(12, 6))

plt.plot(est_t, est_v, label='Vx GTSAM (Estymowana)', color='blue', linewidth=2)
plt.plot(t_imu, v_imu, label='Vx GPS referencyjna)', 
         color='green', alpha=0.5, linestyle='--')

plt.title('Porównanie prędkości estymowanej i referencyjnej')
plt.xlabel('Czas [s]')
plt.ylabel('Prędkość [m/s]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
