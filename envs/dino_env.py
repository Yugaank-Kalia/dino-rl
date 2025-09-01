# envs/dino_pixel_env_selenium.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import json
import os

class DinoEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, config_path=os.path.join('configs', 'game_url.json'), size=(84, 84)):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(3) # 0: Nothing, 1: Jump, 2: Duck
        # The observation space is now a single grayscale image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size[0], size[1], 1), dtype=np.uint8
        )
        self.is_ducking = False

        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.url = cfg.get('url', 'http://localhost:8000')

        chrome_options = Options()
        chrome_options.add_argument('--window-size=800,400')
        chrome_options.add_argument('--disable-application-cache')
        chrome_options.add_argument('--disk-cache-size=0')
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        self.driver.get(self.url)
        time.sleep(1.0)
        self.body = self.driver.find_element(By.TAG_NAME, 'body')

    def _is_game_over(self) -> bool:
        try:
            return self.driver.execute_script("return Runner.instance_.crashed;")
        except Exception:
            return True

    def _is_on_ground(self) -> bool:
        try:
            return not self.driver.execute_script("return Runner.instance_.tRex.jumping;")
        except Exception:
            return False

    def _get_observation(self):
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, 'canvas')
            png = element.screenshot_as_png
            pil_image = Image.open(BytesIO(png))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_frame = clahe.apply(frame)
            processed_frame = cv2.resize(contrast_frame, self.size, interpolation=cv2.INTER_AREA)
            # Add a channel dimension for the CNN
            return np.expand_dims(processed_frame, axis=-1).astype(np.uint8)
        except Exception:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._is_game_over():
            self.body.send_keys(' ')
            time.sleep(0.1)
        self.is_ducking = False
        return self._get_observation(), {}

    def step(self, action: int):
        try:
            on_ground = self._is_on_ground()
            if action == 1 and on_ground:
                if self.is_ducking:
                    self.driver.execute_script("Runner.instance_.tRex.setDuck(false)")
                    self.is_ducking = False
                self.driver.execute_script("Runner.instance_.tRex.startJump(Runner.instance_.currentSpeed)")
            elif action == 2:
                if on_ground and not self.is_ducking:
                    self.driver.execute_script("Runner.instance_.tRex.setDuck(true)")
                    self.is_ducking = True
            else: # action == 0
                if self.is_ducking:
                    self.driver.execute_script("Runner.instance_.tRex.setDuck(false)")
                    self.is_ducking = False
        except Exception:
            pass

        time.sleep(1/15)
        obs = self._get_observation()
        terminated = self._is_game_over()
        reward = -10.0 if terminated else 1.0
        return obs, reward, terminated, False, {}

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass