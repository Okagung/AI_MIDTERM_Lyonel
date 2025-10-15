import os
import time
from rocket_env import SimpleRocketEnv
from reward_wrapper import CustomRewardWrapper
from stable_baselines3 import DQN

# --- Pilih Model yang Ingin Diuji ---
# Ganti nama file sesuai model yang ingin Anda lihat
MODEL_NAME = "DQN_model.zip"  # Pilihan: "DQN_model.zip", "DoubleDQN_model.zip", "DuelingDQN_model.zip"
MODELS_DIR = "../a_tier_models"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# --- Muat Environment dan Model ---
# render_mode='human' wajib untuk melihat visualnya
env = CustomRewardWrapper(SimpleRocketEnv(render_mode='human'))
model = DQN.load(MODEL_PATH, env=env)

# --- Jalankan Satu Episode ---
obs, info = env.reset()
done = False
truncated = False
total_reward = 0

print(f"Menguji model: {MODEL_NAME}")

while not done and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(0.02) # Sedikit jeda agar tidak terlalu cepat

print(f"Episode selesai! Total Reward: {total_reward}")
env.close()