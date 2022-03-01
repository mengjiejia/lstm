import pandas as pd
from datetime import datetime
from functools import reduce
from scipy import stats

def format_my_nanos(nanos):
    dt = datetime.fromtimestamp(nanos / 1e9)
    return '{}{:03.0f}'.format(dt.strftime('%Y-%m-%dT%H:%M:%S.%f'), nanos % 1e3)


def to_ns(df):
    for i in range(len(df['field.header.stamp'])):
        print(format_my_nanos(df['field.header.stamp'][i]))


df_roll = pd.read_csv(
    '/home/mengjie/processed/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure-mavros-nav_info-roll.csv')

df_pitch = pd.read_csv(
    '/home/mengjie/processed/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure-mavros-nav_info-pitch.csv')

df_yaw = pd.read_csv(
    '/home/mengjie/processed/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure-mavros-nav_info-yaw.csv')

df_imu = pd.read_csv(
    '/home/mengjie/processed/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure-mavros-imu-data_raw.csv')

df_airspeed = pd.read_csv(
    '/home/mengjie/processed/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure/carbonZ_2018-10-05-14-37-22_3_left_aileron_failure-mavros-nav_info-airspeed.csv')


def timestamp2sec(df):
    df1 = pd.to_datetime(df['field.header.stamp'], unit='ns')
    print(df1)
    for i in range(len(df['field.header.stamp'])):
        t = df1[i] - df1[0]
        t = t.total_seconds()
        df['field.header.stamp'] = df['field.header.stamp'].astype('float64')
        df['field.header.stamp'][i] = t



roll = []
pitch = []
yaw = []
imu = []
airspeed = []

roll = df_roll[['field.header.stamp','field.commanded','field.measured']]
roll['field.header.stamp'] = pd.to_datetime(df_roll['field.header.stamp'], unit='ns')
# print(df)
roll = roll.resample('100L',on='field.header.stamp', label='right').mean()
roll = roll.fillna(method='ffill')
print(roll)
roll = roll.rename(columns={"field.measured": "roll_measured", "field.commanded": "roll_commanded"})


pitch = df_pitch[['field.header.stamp','field.commanded','field.measured']]
pitch['field.header.stamp'] = pd.to_datetime(df_pitch['field.header.stamp'], unit='ns')
# print(df)
pitch = pitch.resample('100L',on='field.header.stamp', label='right').mean()
pitch = pitch.fillna(method='ffill')
# print(roll)
pitch = pitch.rename(columns={"field.measured": "pitch_measured", "field.commanded": "pitch_commanded"})

yaw = df_yaw[['field.header.stamp','field.commanded','field.measured']]
yaw['field.header.stamp'] = pd.to_datetime(df_yaw['field.header.stamp'], unit='ns')
# print(df)
yaw = yaw.resample('100L',on='field.header.stamp', label='right').mean()
yaw = yaw.fillna(method='ffill')
# print(roll)
yaw = yaw.rename(columns={"field.measured": "yaw_measured", "field.commanded": "yaw_commanded"})

airspeed = df_airspeed[['field.header.stamp','field.commanded','field.measured']]
airspeed['field.header.stamp'] = pd.to_datetime(df_airspeed['field.header.stamp'], unit='ns')
# print(df)
airspeed = airspeed.resample('100L',on='field.header.stamp', label='right').mean()
airspeed = airspeed.fillna(method='ffill')
# print(roll)
airspeed = airspeed.rename(columns={"field.measured": "airspeed_measured", "field.commanded": "airspeed_commanded"})


imu = df_imu[['field.header.stamp','field.angular_velocity.x','field.angular_velocity.y','field.angular_velocity.z']]
imu['field.header.stamp'] = pd.to_datetime(df_imu['field.header.stamp'], unit='ns')
# timestamp = imu['field.header.stamp']
# timestamp = pd.DataFrame(imu['field.header.stamp'])

# print(df)
print("=======================================")
# print(imu)
imu1 = imu.resample('100L',on='field.header.stamp', label='right').mean()
imu2 = imu1.fillna(method='ffill')
imu2 = imu2.rename(columns={"field.angular_velocity.x": "angular_velocity.x", "field.angular_velocity.y": "angular_velocity.y", "field.angular_velocity.z": "angular_velocity.z"})

print("=======================================")
#print(imu1)
print("=======================================")
# print(imu2)
# m = imu2.merge(roll, how='left', on='field.header.stamp')
# m = m.merge(roll, how='left', on='field.header.stamp')

data_frames = [roll, pitch, yaw, airspeed, imu2]
df_merged = reduce(lambda left,right: pd.merge(left,right,on=['field.header.stamp'],
                                            how='outer'), data_frames)
timestamp = pd.DataFrame(df_merged.index)


df_merged.insert (0, "timestamp", timestamp)
print(df_merged)
# df_merged.to_csv('training_data.csv',index=False)



# for i in range(len(df_roll['field.header.stamp'])):
    # roll.append(format_my_nanos(df_roll['field.header.stamp'][i]))

# for i in range(len(df_imu['field.header.stamp'])):
    # imu.append(format_my_nanos(df_imu['field.header.stamp'][i]))

# for i in range(len(df_airspeed['field.header.stamp'])):
    # airspeed.append(format_my_nanos(df_airspeed['field.header.stamp'][i]))

# for item in roll:
#     print(item)

# print("=======================================")

# for item in imu:
#     print(item)

# print("=======================================")

# for item in airspeed:
#    print(item)

# print("=======================================")

# print(roll[0])
# print(imu[0])
# print(airspeed[0])
# print("=======================================")
# print(len(roll))
# print(len(imu))
# print(len(airspeed))
