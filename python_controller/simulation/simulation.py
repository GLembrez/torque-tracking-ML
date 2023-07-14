import time

import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('/home/gabinlembrez/GitHub/torque-tracking-ML/python_controller/simulation/gen3_7dof_mujoco.xml')
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data, title='Kinova Gen3 7 dofs')

# simulate and render
while True:
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
