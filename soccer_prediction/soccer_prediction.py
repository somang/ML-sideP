import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

pd.set_option('display.max_columns', 100)

players = pd.read_csv('complete.csv')
#print(list(players))
"""
'ID', 'name', 'full_name', 
'club', 'club_logo', 'special', 
'age', 'league', 
'birth_date', 'height_cm', 'weight_kg', 
'body_type', 'real_face', 'flag', 'nationality', 
'photo', 'eur_value', 'eur_wage', 'eur_release_clause', 
'overall', 'potential', 'pac', 'sho', 'pas', 'dri', 
'def', 'phy', 'international_reputation', 'skill_moves', 
'weak_foot', 'work_rate_att', 'work_rate_def', 
'preferred_foot', 'crossing', 'finishing', 
'heading_accuracy', 'short_passing', 'volleys', 
'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 
'ball_control', 'acceleration', 'sprint_speed', 'agility', 
'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 
'strength', 'long_shots', 'aggression', 'interceptions', 
'positioning', 'vision', 'penalties', 'composure', 'marking', 
'standing_tackle', 'sliding_tackle', 'gk_diving', 
'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes', 
'rs', 'rw', 'rf', 'ram', 'rcm', 'rm', 'rdm', 'rcb', 'rb', 
'rwb', 'st', 'lw', 'cf', 'cam', 'cm', 'lm', 'cdm', 'cb', 
'lb', 'lwb', 'ls', 'lf', 'lam', 'lcm', 'ldm', 'lcb', 'gk', 
'1_on_1_rush_trait', 'acrobatic_clearance_trait', 
'argues_with_officials_trait', 'avoids_using_weaker_foot_trait',
 'backs_into_player_trait', 'bicycle_kicks_trait', 
 'cautious_with_crosses_trait', 'chip_shot_trait', 
 'chipped_penalty_trait', 'comes_for_crosses_trait', 
 'corner_specialist_trait', 'diver_trait', 
 'dives_into_tackles_trait', 'diving_header_trait', 
 'driven_pass_trait', 'early_crosser_trait', 
 "fan's_favourite_trait", 'fancy_flicks_trait',
  'finesse_shot_trait', 'flair_trait', 'flair_passes_trait', 
  'gk_flat_kick_trait', 'gk_long_throw_trait', 
  'gk_up_for_corners_trait', 'giant_throw_in_trait', 
  'inflexible_trait', 'injury_free_trait', 
  'injury_prone_trait', 'leadership_trait', 
  'long_passer_trait', 'long_shot_taker_trait', 
  'long_throw_in_trait', 'one_club_player_trait', 
  'outside_foot_shot_trait', 'playmaker_trait', 
  'power_free_kick_trait', 'power_header_trait', 
  'puncher_trait', 'rushes_out_of_goal_trait', 
  'saves_with_feet_trait', 'second_wind_trait', 
  'selfish_trait', 'skilled_dribbling_trait', 
  'stutter_penalty_trait', 'swerve_pass_trait', 
  'takes_finesse_free_kicks_trait', 'target_forward_trait', 
  'team_player_trait', 'technical_dribbler_trait', 
  'tries_to_beat_defensive_line_trait', 'poacher_speciality', 
  'speedster_speciality', 'aerial_threat_speciality', 
  'dribbler_speciality', 'playmaker_speciality', 
  'engine_speciality', 'distance_shooter_speciality', 
  'crosser_speciality', 'free_kick_specialist_speciality', 
  'tackling_speciality', 'tactician_speciality', 
  'acrobat_speciality', 'strength_speciality', 
  'clinical_finisher_speciality', 'prefers_rs', 
  'prefers_rw', 'prefers_rf', 'prefers_ram', 'prefers_rcm', 
  'prefers_rm', 'prefers_rdm', 'prefers_rcb', 'prefers_rb', 
  'prefers_rwb', 'prefers_st', 'prefers_lw', 'prefers_cf',
   'prefers_cam', 'prefers_cm', 'prefers_lm', 'prefers_cdm', 
   'prefers_cb', 'prefers_lb', 'prefers_lwb', 'prefers_ls', 
   'prefers_lf', 'prefers_lam', 'prefers_lcm', 'prefers_ldm', 
   'prefers_lcb', 'prefers_gk'
"""

#print(players[['full_name', 'age', 'nationality', 'overall', 'potential']])

compt = pd.read_csv('results.csv') #competitions
print(list(compt))
"""
['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'city', 'country', 'neutral']

"""

# given home team-away team, predict scores
######################## prepare the data ########################
train, test = train_test_split(compt, test_size=0.2)
train_x, train_y = pd.Categorical(train[['home_team', 'away_team']]), train[['home_score', 'away_score']]
test_x, test_y = pd.Categorical(test[['home_team', 'away_team']]), test[['home_score', 'away_score']]

######################## set learning variables ##################
learning_rate = 0.0005
epochs = 2000
batch_size = 3

######################## set some variables #######################
x = tf.placeholder(tf.float32, [None, 2], name='x')  # 2 features
y = tf.placeholder(tf.float32, [None, 2], name='y')  # 2 outputs

# hidden layer 1
W1 = tf.Variable(tf.truncated_normal([3, 10], stddev=0.03), name='W1')
b1 = tf.Variable(tf.truncated_normal([10]), name='b1')

# hidden layer 2
W2 = tf.Variable(tf.truncated_normal([10, 3], stddev=0.03), name='W2')
b2 = tf.Variable(tf.truncated_normal([3]), name='b2')

'''
######################## Activations, outputs ######################
# output hidden layer 1
hidden_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

# total output
y_ = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))

####################### Loss Function  #########################
mse = tf.losses.mean_squared_error(y, y_)

####################### Optimizer      #########################
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

###################### Initialize, Accuracy and Run #################
# initialize variables
init_op = tf.global_variables_initializer()

# accuracy for the test set
accuracy = tf.reduce_mean(tf.square(tf.subtract(y, y_)))  # or could use tf.losses.mean_squared_error

# run
with tf.Session() as sess:
  writer = tf.summary.FileWriter('logs', sess.graph)

  sess.run(init_op)
  total_batch = int(len(y_train) / batch_size)
  for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
      batch_x, batch_y = X_train[i * batch_size:min(i * batch_size + batch_size, len(X_train)), :], \
                         y_train[i * batch_size:min(i * batch_size + batch_size, len(y_train)), :]
      _, c = sess.run([optimizer, mse], feed_dict={x: batch_x, y: batch_y})
      avg_cost += c / total_batch
    if epoch % 10 == 0:
      print 'Epoch:', (epoch + 1), 'cost =', '{:.3f}'.format(avg_cost)
  print sess.run(mse, feed_dict={x: X_test, y: y_test})

  writer.close()

'''