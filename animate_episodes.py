import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax
import jax.numpy as jnp
from tqdm import tqdm
import pickle
from save_q_reps import load_checkpoint
import os
# os.environ["MUJOCO_GL"] = "egl"  # or "osmesa"
# # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["MUJOCO_EGL_DEVICE_ID"] = "2"  # force use of GPU 2 only
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# import os
# os.environ["MUJOCO_GL_VERBOSE"] = "1"

# import os
# os.environ["MUJOCO_GL_VERBOSE"] = "1"

# import os
# os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
# # os.environ["EGL_VISIBLE_DEVICES"] = "2"

import os
# Disable hardware acceleration
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["EGL_DEVICE_ID"] = ""

# import mujoco
# print(mujoco.__version__)  # should be something like 2.3.7
# # import mujoco
# # print(mujoco.get_rendering_backend())
# import os
# os.environ["MUJOCO_GL"] = "egl"  # Try egl first
# import mujoco
# model = mujoco.MjModel.from_xml_string("""
# <mujoco>
#   <worldbody>
#     <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
#   </worldbody>
# </mujoco>
# """)
# data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model)
# renderer.update_scene(data)
# pixels = renderer.render()
# print("✅ Rendered an image with shape:", pixels.shape)




# (Assume all your previous simulation and load_checkpoint code is available.)

def create_episode_video(ckpt_num, num_episodes=1, image_height=480, image_width=640,
                           save_path=None, alpha='0.1', misc_params='0.1_None',
                           env_name='sawyer_bin', seed=1):
    """
    Loads the checkpoint corresponding to ckpt_num, runs one or more episodes in the environment,
    collects images rendered using the "gripperPOV" mode, and saves a video of the episode(s).
    
    Args:
        ckpt_num (int): Checkpoint number to load.
        num_episodes (int): Number of episodes to simulate (default: 1).
        image_height (int): Height of the rendered image.
        image_width (int): Width of the rendered image.
        save_path (str): File path to save the video (if None, defaults to "./plots/episode_ckpt_{ckpt_num}.mp4").
        alpha (str): Hyperparameter string for load_checkpoint.
        misc_params (str): Miscellaneous parameters.
        env_name (str): Name of the environment.
        seed (int): Random seed.
    """
    from acme.jax import utils  # if needed
    # Use the existing load_checkpoint function to get the trained learner, env, and networks.
    trained_learner_state, env, networks = load_checkpoint(alpha, misc_params, env_name, base_log_dir=None, seed=seed, fix_goals=True, ckpt_num=ckpt_num)
    
    # Define a JIT-ed policy selection function
    @jax.jit
    def select_action(params, obs):
        dist = networks.policy_network.apply(params, obs)
        return dist.mode()
    
    # We'll collect images for each episode in a list.
    episode_images = []  # we'll only use the first episode's images to create the video
    for epi in range(num_episodes):
        env.seed(np.random.randint(1e6))
        timestep = env.reset()
        images = []
        print("hi")
        while not timestep.last():
            # Render an image from the environment using "gripperPOV" mode.
            # Adjust mode parameters if needed.
            os.environ["MUJOCO_GL"] = "osmesa"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["EGL_DEVICE_ID"] = ""
            #img = env.render()
            print("Available cameras:", env.sim.model.camera_names)

            print("hi i am here")
            img = env.sim.render(width=640, height=480, camera_name="topview")
            #img = env.render(mode="rgb_array", width=640, height=480)
            #img = env.sim.render(width=320, height=240, camera_name="topview")
            
            img = 1
            images.append(img)
            print("hiiiiii")
            # Select action using the policy.
            action = np.array(select_action(trained_learner_state.policy_params, timestep.observation))
            # Step the environment.
            timestep = env.step(action)
            #env.render("rgb_array")
        
        # For video, we'll use the images from the first episode.
        if epi == 0:
            episode_images = images
            
        print(f"Episode {epi} return: {timestep.reward}")
    
    # Convert the list of images into a numpy array
    imgs = np.array(episode_images)  # shape: (num_frames, H, W, 3)
    
    # Create the animation
    fig = plt.figure()
    im = plt.imshow(imgs[0])
    
    def init():
        im.set_data(imgs[0])
        return [im]
    
    def animate(i):
        im.set_data(imgs[i])
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=imgs.shape[0], interval=100)
    
    if save_path is None:
        save_path = f"./plots/episode_ckpt_{ckpt_num}.mp4"
    # Save the animation as an MP4 file.
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(save_path, writer=FFwriter)
    plt.close()
    print(f"✅ Video saved to '{save_path}'")
    return save_path


# Example usage:
if __name__ == "__main__":
    create_episode_video(ckpt_num=15, num_episodes=1)