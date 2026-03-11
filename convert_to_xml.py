import mujoco

model = mujoco.MjModel.from_xml_path("dummy.urdf")
mujoco.mj_saveLastXML("dummy.xml", model)