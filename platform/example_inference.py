import torch
from rocket_tvc import Rocket
from policy_tvc import ActorCritic
import os
import glob


class RocketInference():

    def __init__(self):
        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.task = 'landing'
        self.max_steps = 1000
        self.ckpt_dir = glob.glob(os.path.join(self.task+'_ckpt', '*.pt'))[-1]  # last ckpt

        self.env = Rocket(task=self.task, max_steps=self.max_steps)
        self.net = ActorCritic(input_dim=self.env.state_dims, output_dim=self.env.action_dims).to(self.device)
        if os.path.exists(self.ckpt_dir):
            checkpoint = torch.load(self.ckpt_dir, weights_only=False)
            self.net.load_state_dict(checkpoint['model_G_state_dict'])
        
        #seed_value = random.randint(1, 7)
        #env.seed(3)

    def run_inference(self, state, init_interference):
        crop_scale=0.4
        self.env.get_wind_info(init_interference)
        action, log_prob, value = self.net.get_action(state)
        state, reward, done, _ = self.env.step(action)
        frame_0, frame_1 = self.env.render()
        frame_0 = self.env.crop_alongwith_camera(frame_0, crop_scale=crop_scale)
        frame_1 = self.env.crop_alongwith_camera(frame_1, crop_scale=crop_scale)

        return frame_0, frame_1, state, done

if __name__ == "__main__":
    inference = RocketInference()
    inference.run_inference()