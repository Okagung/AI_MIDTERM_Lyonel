import gymnasium as gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    """
    Wrapper ini memodifikasi reward untuk mendorong perilaku yang diinginkan:
    - Mendekati target dengan aman.
    - Mendarat dalam posisi tegak.
    - Tidak bergerak terlalu cepat.
    """
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def step(self, action):
        # Jalankan satu step di environment asli
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ambil state yang sudah dinormalisasi kembali ke nilai aslinya
        # PERBAIKAN DI SINI: Menggunakan self.env._normalizer()
        state_values = obs * self.env._normalizer()
        x, y, vx, vy, sin_theta, cos_theta, omega, dx, dy = state_values

        # --- Desain Reward Shaping ---

        # 1. Reward karena mendekati target (semakin kecil jarak, semakin baik)
        distance_to_target = np.sqrt(dx**2 + dy**2)
        reward = -0.01 * distance_to_target

        # 2. Penalti untuk kecepatan tinggi (mendorong pendaratan lembut)
        speed = np.sqrt(vx**2 + vy**2)
        reward -= 0.005 * speed

        # 3. Penalti untuk posisi miring (mendorong pendaratan tegak)
        reward += 0.005 * (cos_theta - 1)

        # 4. Penalti/Reward besar untuk kondisi terminal
        if terminated:
            # PERBAIKAN DI SINI: Mengakses atribut dari self.env
            tx, ty = self.env.target_pos
            half_w, half_h = self.env.target_w / 2, self.env.target_h / 2
            is_on_target = (tx - half_w <= x <= tx + half_w) and (ty - half_h <= y <= ty + half_h)

            if is_on_target:
                # 5. Reward besar untuk pendaratan sukses!
                print("Landed on target! GREAT SUCCESS!")
                reward = 200.0
            else:
                # Penalti besar jika jatuh di luar target
                print("Crashed or went out of bounds.")
                reward = -100.0

        return obs, reward, terminated, truncated, info

    # Catatan: Fungsi _normalizer() tidak lagi dibutuhkan di wrapper
    # karena kita bisa memanggilnya langsung dari self.env._normalizer()