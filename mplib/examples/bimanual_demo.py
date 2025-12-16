#!/usr/bin/env python3

import numpy as np
import sapien.core as sapien
import os
from mplib.examples.bimanual_demo_setup import BimanualDemoSetup


class BimanualPlanningDemo(BimanualDemoSetup):
    """
    Bimanual motion planning demo where both robots move independently and collaboratively.
    - Phase 1: Both arms move to independent target poses
    - Phase 2: Both arms approach and manipulate objects
    - Phase 3 (optional): Both arms hold a common object (constrained grasp)
    """

    def __init__(self):
        """
        Setup the scene, load the dual-arm robot, and initialize the planner.
        """
        super().__init__()
        # Load the world, the robot, and then setup the planner
        self.setup_scene()
        self.load_robot()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_assets_dir = os.path.join(current_dir, "panda_assets")
        srdf_path = os.path.join(self.local_assets_dir, "mobile_panda_dual_arm_fixed.srdf")
        # srdf_path = ""
        self.setup_planner(srdf_path=srdf_path)

        # Check the actual DOF of the loaded robot
        print(f"Robot DOF: {self.robot.dof}")
        init_qpos = np.zeros(self.robot.dof)
        # 1. Base joints (0-3)
        # Keep X=0.0 to stay safely away from the table at X=0.6
        init_qpos[0:4] = [0.0, 0.0, 0.0, 0]

        # 2. Right arm (4-10) - Standard "Home" Pose
        init_qpos[4:11] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        # Alternative (The one you used):
        # init_qpos[4:11] = [0, 0.19, 0.0, -2.62, 0.0, 2.94, 0.79]

        # 3. Right Gripper (11-12) - Open
        # Note: Gripper indices are 11 and 12
        init_qpos[11:13] = [0.04, 0.04]

        # 4. Left arm (13-19)
        # Note: Left Arm starts at index 13 (4 base + 7 arm + 2 gripper)
        # We MIRROR the pose (invert Joint 1, 3, 5, 7) for symmetry
        init_qpos[13:20] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        # 5. Left Gripper (20-21) - Open
        init_qpos[20:22] = [0.04, 0.04]
        
        # Apply and Update
        self.robot.set_qpos(init_qpos)
        self.active_joints = self.robot.get_active_joints()
        for i, joint in enumerate(self.active_joints):
            joint.set_drive_target(init_qpos[i])
        # Create table for objects to rest on
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.5, 0.5, 0.025])
        builder.add_box_visual(half_size=[0.5, 0.5, 0.025])
        table = builder.build_kinematic(name="table")
        table.set_pose(sapien.Pose([0.6, 0, -0.025]))

        # Create two boxes for bimanual manipulation
        # Left box (for left arm)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.03, 0.03, 0.05])
        builder.add_box_visual(half_size=[0.03, 0.03, 0.05])
        self.left_box = builder.build(name="left_box")
        self.left_box.set_pose(sapien.Pose([0.3, 0.2, 0.05]))

        # Right box (for right arm)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.03, 0.03, 0.05])
        builder.add_box_visual(half_size=[0.03, 0.03, 0.05])
        self.right_box = builder.build(name="right_box")
        self.right_box.set_pose(sapien.Pose([0.6, -0.2, 0.05]))

        # Optional: Create a shared object for constrained grasp demo
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.04, 0.02, 0.08])
        builder.add_box_visual(half_size=[0.04, 0.02, 0.08])
        self.shared_object = builder.build(name="shared_object")
        self.shared_object.set_pose(sapien.Pose([0.45, 0, 0.08]))

    # def get_local_target(robot, arm_base_link_index, world_target_pose):
    #     # 1. Get the Arm's Base Pose in World Frame
    #     # (You need to find the correct link index for the shoulder/base of the arm)
    #     T_world_base = robot.get_links()[arm_base_link_index].get_pose()
        
    #     # 2. Compute the Local Target: T_local = T_base^(-1) * T_world
    #     T_target_world = sapien.Pose(world_target_pose[:3], world_target_pose[3:])
    #     T_target_local = T_world_base.inv() * T_target_world
        
    #     # Return as list [x, y, z, w, x, y, z] (Check quaternion order below!)
    #     p = T_target_local.p
    #     q = T_target_local.q
    #     return [p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
    
    def demo_independent_motion(self):
        """
        Phase 1: Both arms move independently to different target poses.
        This demonstrates basic bimanual motion planning without interaction.
        """
        print("\n=== Phase 1: Independent Motion ===")
        print("Moving left arm to target pose and right arm to another pose simultaneously...")

        # Define target poses for each arm
        # Left arm target: [x, y, z, qx, qy, qz, qw]
        left_target = [0.3, 0.2, 0.2, 0, 0, 0, 1]
        
        # Right arm target: [x, y, z, qx, qy, qz, qw]
        right_target = [0.6, -0.2, 0.2, 0, 0, 0, 1]
        print(self.robot)
        # Plan and move both arms to their targets
        result = self.move_to_pose_pair(left_target, right_target)
        if result == -1:
            print("Failed to plan independent motion")
            return False
        print("✓ Independent motion completed")
        return True

    def demo_synchronized_pickup(self):
        """
        Phase 2: Both arms approach, pick up boxes, and move them together.
        Demonstrates synchronized bimanual motion.
        """
        print("\n=== Phase 2: Synchronized Pickup ===")
        
        # Left arm: approach left box
        print("Left arm approaching box...")
        left_approach = [0.3, 0.2, 0.15, 0, 1, 0, 0]
        left_grasp = [0.3, 0.2, 0.05, 0, 1, 0, 0]
        
        # Move to approach position
        right_approach = [0.6, -0.2, 0.15, 0, 1, 0, 0]
        result = self.move_to_pose_pair(left_approach, right_approach)
        if result == -1:
            print("Failed to reach approach position")
            return False
        print("✓ Both arms at approach position")

        # Move down to grasp
        print("Moving down to grasp...")
        right_grasp = [0.6, -0.2, 0.05, 0, 1, 0, 0]
        result = self.move_to_pose_pair(left_grasp, right_grasp)
        if result == -1:
            print("Failed to reach grasp position")
            return False
        print("✓ Both arms at grasp position")

        # Close grippers
        print("Closing grippers...")
        self.close_both_grippers()
        print("✓ Objects grasped")

        # Lift objects
        print("Lifting objects...")
        left_lift = [0.3, 0.2, 0.25, 0, 1, 0, 0]
        right_lift = [0.6, -0.2, 0.25, 0, 1, 0, 0]
        result = self.move_to_pose_pair(left_lift, right_lift)
        if result == -1:
            print("Failed to lift objects")
            return False
        print("✓ Objects lifted")

        # Move to new positions (exchange positions)
        print("Moving objects to new positions...")
        left_new = [0.4, -0.15, 0.25, 0, 1, 0, 0]
        right_new = [0.5, 0.15, 0.25, 0, 1, 0, 0]
        result = self.move_to_pose_pair(left_new, right_new)
        if result == -1:
            print("Failed to move to new positions")
            return False
        print("✓ Objects at new positions")

        # Place objects down
        print("Placing objects down...")
        left_place = [0.4, -0.15, 0.05, 0, 1, 0, 0]
        right_place = [0.5, 0.15, 0.05, 0, 1, 0, 0]
        result = self.move_to_pose_pair(left_place, right_place)
        if result == -1:
            print("Failed to place objects")
            return False

        # Open grippers
        print("Opening grippers...")
        self.open_both_grippers()
        print("✓ Objects placed and released")
        return True

    def demo_symmetric_motion(self):
        """
        Phase 3: Both arms move symmetrically (mirror motion).
        Useful for tasks that have left-right symmetry.
        """
        print("\n=== Phase 3: Symmetric Motion ===")
        print("Both arms moving symmetrically...")

        # Define a center position
        center_x = 0.45
        center_y = 0.0
        center_z = 0.2

        # Left arm moves to offset on one side
        left_pose = [center_x - 0.15, center_y + 0.15, center_z, 0, 1, 0, 0]
        
        # Right arm moves to mirror position
        right_pose = [center_x + 0.15, center_y - 0.15, center_z, 0, 1, 0, 0]

        result = self.move_to_pose_pair(left_pose, right_pose)
        if result == -1:
            print("Failed to perform symmetric motion")
            return False
        print("✓ Symmetric motion completed")
        return True

    def demo(self):
        """Run the complete bimanual demo"""
        print("\n" + "=" * 60)
        print("BIMANUAL ROBOT MOTION PLANNING DEMO")
        print("=" * 60)
        self.lock_base_joints()
        # Open grippers initially
        print("Opening both grippers...")
        self.open_both_grippers()
        collisions = self.planner.check_for_env_collision(qpos=self.robot.get_qpos())
        if len(collisions) > 0:
            print("HI 6")
            print(f"❌ STILL COLLIDING! Found {len(collisions)} contacts:")
            for c in collisions:
                 # Try accessing different attributes depending on MPLib version
                 n1 = getattr(c, "link_name1", getattr(c, "object_name1", str(c)))
                 n2 = getattr(c, "link_name2", getattr(c, "object_name2", str(c)))
                 print(f"  {n1} <--> {n2}")
            return
        print("HI 7")
        # Phase 1: Independent motion
        if not self.demo_independent_motion():
            print("Demo stopped at Phase 1")
            return

        # Phase 2: Synchronized pickup
        if not self.demo_synchronized_pickup():
            print("Demo stopped at Phase 2")
            return

        # Phase 3: Symmetric motion
        if not self.demo_symmetric_motion():
            print("Demo stopped at Phase 3")
            return

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)


if __name__ == "__main__":
    demo = BimanualPlanningDemo()
    demo.demo()
