import os
from rocket_env import SimpleRocketEnv
from reward_wrapper import CustomRewardWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# --- Konfigurasi ---
MODELS_DIR = "../a_tier_models"
LOGS_DIR = "../logs"
TOTAL_TIMESTEPS = 200000  # Jumlah langkah pelatihan, bisa dinaikkan untuk hasil lebih baik

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# --- Buat Environment dengan Reward Kustom ---
# render_mode='human' agar bisa dilihat, tapi pelatihan lebih cepat tanpa render
env = CustomRewardWrapper(SimpleRocketEnv(render_mode=None))
env.reset()

# --- Definisi Model untuk A-Tier ---
models_to_train = {
    "DQN": {
        "model": DQN,
        "params": {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.02,
            "verbose": 1,
            "tensorboard_log": LOGS_DIR
        }
    },
    "DoubleDQN": {
        "model": DQN,
        "params": {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.02,
            "double_q": True,  # Ini yang mengaktifkan Double DQN
            "verbose": 1,
            "tensorboard_log": LOGS_DIR
        }
    },
    "DuelingDQN": {
        "model": DQN,
        "params": {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.02,
            "policy_kwargs": dict(dueling=True),  # Ini yang mengaktifkan Dueling DQN
            "verbose": 1,
            "tensorboard_log": LOGS_DIR
        }
    }
}

# --- Lakukan Pelatihan untuk Setiap Model ---
for name, config in models_to_train.items():
    print(f"==========================================")
    print(f"           TRAINING: {name}               ")
    print(f"==========================================")
    
    ModelClass = config["model"]
    params = config["params"]
    
    model = ModelClass(**params)
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=name,
        progress_bar=True
    )
    
    # Simpan model yang sudah dilatih
    save_path = os.path.join(MODELS_DIR, f"{name}_model.zip")
    model.save(save_path)
    print(f"Model {name} disimpan di: {save_path}")

env.close()

print("\n\nSemua pelatihan selesai!")