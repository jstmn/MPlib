#!/usr/bin/env python3

import numpy as np
import sapien.core as sapien

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
        self.setup_planner()

        # Set initial joint positions (all arms in a neutral pose)
        # Note: Adjust these based on your specific URDF joint ordering
        init_qpos = np.zeros(17)  # Adjust based on your robot's DOF
        # Base joints (mobile base): root_x, root_y, root_z_rotation, height
        init_qpos[0] = 0.0   # root_x_axis_joint
        init_qpos[1] = 0.0   # root_y_axis_joint
        init_qpos[2] = 0.0   # root_z_rotation_joint
        init_qpos[3] = 0.0   # linear_actuator_height

        # Right arm (7 DOF Panda): joints 4-10
        init_qpos[4:11] = [0, 0.19, 0.0, -2.62, 0.0, 2.94, 0.79]
        
        # Left arm (7 DOF Panda): joints 11-17
        init_qpos[11:18] = [0, 0.19, 0.0, -2.62, 0.0, 2.94, 0.79]
        
        self.robot.set_qpos(init_qpos)

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
        builder.add_box_visual(half_size=[0.03, 0.03, 0.05], color=[1, 0, 0])
        self.left_box = builder.build(name="left_box")
        self.left_box.set_pose(sapien.Pose([0.3, 0.2, 0.05]))

        # Right box (for right arm)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.03, 0.03, 0.05])
        builder.add_box_visual(half_size=[0.03, 0.03, 0.05], color=[0, 1, 0])
        self.right_box = builder.build(name="right_box")
        self.right_box.set_pose(sapien.Pose([0.6, -0.2, 0.05]))

        # Optional: Create a shared object for constrained grasp demo
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.04, 0.02, 0.08])
        builder.add_box_visual(half_size=[0.04, 0.02, 0.08], color=[0, 0, 1])
        self.shared_object = builder.build(name="shared_object")
        self.shared_object.set_pose(sapien.Pose([0.45, 0, 0.08]))

    def demo_independent_motion(self):
        """
        Phase 1: Both arms move independently to different target poses.
        This demonstrates basic bimanual motion planning without interaction.
        """
        print("\n=== Phase 1: Independent Motion ===")
        print("Moving left arm to target pose and right arm to another pose simultaneously...")

        # Define target poses for each arm
        # Left arm target: [x, y, z, qx, qy, qz, qw]
        left_target = [0.3, 0.2, 0.2, 0, 1, 0, 0]
        
        # Right arm target: [x, y, z, qx, qy, qz, qw]
        right_target = [0.6, -0.2, 0.2, 0, 1, 0, 0]

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

        # Open grippers initially
        print("Opening both grippers...")
        self.open_both_grippers()

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
