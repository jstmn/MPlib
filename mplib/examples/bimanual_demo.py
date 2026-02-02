#!/usr/bin/env python3

import numpy as np
import sapien.core as sapien
from mplib.examples.bimanual_demo_setup import BimanualDemoSetup_v3
import time

class BimanualPlanningDemo_v3(BimanualDemoSetup_v3):
    def __init__(self):
        super().__init__()
        self.setup_scene()
        self.load_robot()
        self.setup_planner()


        print(f"Robot DOF: {self.robot.dof}")
        
        # CHANGED: 18 DOF Initialization (No Mobile Base)
        # Order: 
        # Panda 1 (Right): 7 Arm + 2 Gripper
        # Panda 2 (Left):  7 Arm + 2 Gripper
        # init_qpos = np.zeros(self.robot.dof)
        
        # # Panda 1 (Right) - Neutral Pose
        # init_qpos[0:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        # init_qpos[7:9] = [0.04, 0.04] # Gripper Open

        # # Panda 2 (Left) - Neutral Pose
        # init_qpos[9:16] = [0, 0.785, 0, -2.356, 0, 1.571, -0.785]
        # init_qpos[16:18] = [0.04, 0.04] # Gripper Open
        init_qpos = np.zeros(self.robot.dof)

        # Panda 1 arm joints (even indices 0,2,4,6,8,10,12)
        panda_1_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, val in enumerate(panda_1_config):
            init_qpos[i*2] = val

        # Panda 2 arm joints (odd indices 1,3,5,7,9,11,13)
        panda_2_config = [3.14, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, val in enumerate(panda_2_config):
            init_qpos[i*2 + 1] = val

        # Grippers
        init_qpos[14:18] = [0.04, 0.04, 0.04, 0.04]  # All grippers open
        # Step the simulation to apply the joint positions
        for _ in range(100):  # Step enough times for joints to reach targets
            qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()

        # Update rendering to show the new pose
        self.scene.update_render()
        self.viewer.render()

        self.robot.set_qpos(init_qpos)
        self.active_joints = self.robot.get_active_joints()
        for i, joint in enumerate(self.active_joints):
            joint.set_drive_target(init_qpos[i])
        print("HI 8")
        # Create objects on table
        # Table is at z=0.8, so objects should be at z ~ 0.825
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.03, 0.03, 0.05])
        builder.add_box_visual(half_size=[0.03, 0.03, 0.05])
        self.left_box = builder.build(name="left_box")
        # Position relative to world (Panda 2 is at y=0.6)
        self.left_box.set_pose(sapien.Pose([0.4, 0.4, 0.85])) 

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.03, 0.03, 0.05])
        builder.add_box_visual(half_size=[0.03, 0.03, 0.05])
        self.right_box = builder.build(name="right_box")
        # Position relative to world (Panda 1 is at y=-0.6)
        self.right_box.set_pose(sapien.Pose([0.4, -0.4, 0.85]))

    def demo_independent_motion(self):
        print("\n=== Phase 1: Independent Motion ===")
        # Panda 2 (Left) Target: Forward and Left
        # [x, y, z, qw, qx, qy, qz]
        left_target = [0.3, 0.2, 1, 0, 1, 0, 0] 
        
        # Panda 1 (Right) Target: Forward and Right
        right_target = [0.3, -0.2, 1, 0, 1, 0, 0]
        
        result = self.move_to_pose_pair(left_target, right_target)
        if result == -1: return False
        return True
    def demo_synchronized_pickup(self):
        print("\n=== Phase 2: Synchronized Pickup ===")
        
        # Approach (Above Boxes)
        left_approach = [0.4, 0.4, 1.0, 0, 1, 0, 0]
        right_approach = [0.4, -0.4, 1.0, 0, 1, 0, 0]
        self.move_to_pose_pair(left_approach, right_approach)
        print("AT (0.4,0.4,1)")
        time.sleep(5)
        # Grasp (At Box Level)
        left_grasp = [0.4, 0.4, 0.85, 0, 1, 0, 0]
        right_grasp = [0.4, -0.4, 0.85, 0, 1, 0, 0]
        self.move_to_pose_pair(left_grasp, right_grasp)
        print("AT",(0.4,0.4,0.85))
        self.close_both_grippers()
        time.sleep(5)
        # Lift
        self.move_to_pose_pair(left_approach, right_approach)
        print("AT",(0.4,0.4,1))
        self.open_both_grippers()
        time.sleep(5)
        return True

    def demo(self):
        self.open_both_grippers()
        
        # Warmup collision check
        collisions = self.planner.check_for_env_collision(qpos=self.robot.get_qpos())
        if len(collisions) > 0:
            print(f"Initial Collision Detected! {len(collisions)} contacts.")
            return
        # if not self.demo_independent_motion(): return
        if not self.demo_independent_motion(): return
        if not self.demo_synchronized_pickup(): return
        print("\nDEMO COMPLETED")

if __name__ == "__main__":
    demo = BimanualPlanningDemo_v3()
    demo.demo()
